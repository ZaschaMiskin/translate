## Input Pipeline

### Anmerkungen
* Das Skript ```batch.py``` bietet die Funktionalität die Daten einzulesen und in das gewünschte (S,T,L)-Format zu überführen.
Zur Einbindung der Daten in den Input des Computational Graph's von TensorFlow müssten diese Daten irgendwie einem 
[```tf.data.Dataset```](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) übergeben werden.
* Beachte, dass ```tf.data.Dataset``` eigens Funktionen bietet die Daten in Batches zusammenzuführen, welche auch
verwendet werden sollten (da ```Dataset``` umfangreichere Funktionalitäten zum Batchhandling bietet als ```batch.py```,
z.B. [shuffling](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle)). Daher sollten ```Dataset``` 
die Trainingsdaten im nicht-gebatchten (S,T,L)-Format übergeben werden. Für das (S,T,L)-Format kann aber ```batch.py```
verwendet werden.


## Modell

#### Anmerkungen zur Implementierung

* Der Computational Graph des Netzwerks sollte in einer Funktion erstellt werden. Bei der Erstellung von häufig auftretenden
Subgraphen (z.B. Layer der gleichen Konfiguration) lohnt es sich evtl. diese auch in Funktionen auszulagern.
* Es bietet sich an die Implementierung des Graphen in ein ```ffnn.py```-File auszulagern.
* Durch TensorFlow's [```tf.nn.embedding_lookup```](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup) 
Funktion werden Indizes (d.h. Skalare) von Wörtern auf embedding-Vektoren gemappt. Bei Eingabe eines 1D-Tensors (Vektor,
welcher Wort-Indizes enthält) erhält man einen 2D-Tensor (Matrix mit embedding-Vektoren als Zeilen). Für die folgenden
Fully Connected Layer wollen wir aber wieder einen 1D-Tensor haben. Genauer möchten wir die embedding-Vektoren (d.h. die
Zeilen des 2D-Tensors) zu einem einzigen feature-Vektor konkatenieren. Dies kann z.B. durch Anwendung der 
[```tf.reshape```](https://www.tensorflow.org/api_docs/python/tf/reshape) Operation erreicht werden.
* Zur Implementierung von Batch Normalization kann TensorFlow's 
[```tf.layers.batch_normalization```](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization) Funktion
verwendet werden. Diese muss dabei vor den Input jedes Fully Connected Layers geschaltet werden (Normalization Layer).
Es ist zu beachten, dass zum Trainieren von Normalization Layer ein paar Anpassungen an der Trainingsprozedur vorgenommen
werden müssen (siehe "Note" in der [Dokumentation](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)).
Zudem müssen dem Graphen des Normalization Layers je nach Anwendungsfall (d.h. ob das Modell trainiert oder eingesetzt
wird) andere Argumente übergeben werden. Für eine detaillierte Erklärung von Batch Normalization, ziehe das Paper [IS15]
zurate (siehe ```../literature/```).
* Später soll das Modell mit dem Cross Entropy Loss trainiert werden. Das Cross Entropy Loss (CE) spielt wunderbar mit
der Softmax Operation (SM) zusammen (u.a. wegen dem ```log``` in CE und dem ```exp``` in SM), daher bietet TensorFlow
zur Vereinfachung der Berechnung eine 
[```tf.losses.softmax_cross_entropy```](https://www.tensorflow.org/api_docs/python/tf/losses/softmax_cross_entropy)
Loss-Funktion, welche SM und CE vereint. Wir brauchen CE aber nur während des Trainings, SM ist jedoch Teil des Modells.
Wir sollten das Modell so implementieren, dass wir während des Traings die Ausgabe des Projektion-Layers der 
```tf.losses.softmax_cross_entropy``` Loss Funktion übergeben. Bei der Anwendung des Netzwerks schalten wir um und übergeben
die Ausgabe des Projektion-Layers der [```tf.nn.softmax```](https://www.tensorflow.org/api_docs/python/tf/nn/softmax)
Funktion welche die Ausgabe lediglich softmax-normalisiert ohne den CE-Loss zu ermitteln.
* Es ist auch nicht zu vergessen einen Anwendungsmodus einzubauen.

#### Entscheidungsfragen

* Welche Fenstergröße ```w``` sollen wir wählen? (w = 2)
* Welche Größe sollen die embedding-Vektoren haben? (vocab_size**0.25)
* Welche Größe sollen die Fully Connected Layer haben? (Sowohl die Layer nach der Konkatenation, als auch davor)
* Wie viele (Fully Connected) Layer wollen wir einbauen?
* Welche Aktivierungsfunktion(-en) sollen wir verwenden? (sigmoid, vorerst)
* Wie sollen die Layer initialisiert werden?
* In der Vorlesung wird gewarnt, dass Batch Normalization die Performance des Modells beeinträchtigen kann. An welchen
Stellen sollen wir denn nun ```tf.layers.batch_normalization``` vorschalten? (vor jedem dense layer (hidden))


## Training

#### Anmerkungen zur Implementierung

* Unter [BL17, S. 33] (siehe ```../literature/```) befindet sich ein Diagramm welches einen Trainingsworkflow vorgibt,
es empfiehlt sich auf dieser Vorgabe aufzubauen. Es lohnt sich das gesamte Kapitel "Test Sets, Validation Sets, and 
Overfitting" zu lesen um Einblicke in Hintergedanken der Datenaufteilung zu erhalten.
* Batch Normalization macht Optimierungsverfahren wie Drop-Out und L2-Regularization überflüssig [BL17, S. 106].
* Beachte die Besonderheiten für das Trainieren von Batch Normalization Layer auf welche schon unter dem Punkt "Modell"
(s.o.) hingewiesen wurde.

#### Anforderungen an die Trainingsroutine

* **Modell Struktur**: Vor Beginn des Trainings sollten die auf dem Aufgabenblatt genannten Modellinfos ausgegeben werden.
* **Metriken**: Während des Trainings sollen über regelmäßige Intervalle die Metriken Accuracy (ACC) und Perplexity (PPL)
über Teile der Trainingsdaten (z.B. über einem Batch) ermittelt und ausgegeben werden.
* **Validierung**: Nach ```n``` Updates ([BL17, S. 33] sagt z.B. nach einer Epoche) sollen die Metriken (insbesondere ACC) über 
den gesamten Entwicklungsdaten (Validierungsdaten) ermittelt und ausgegeben werden.
* **Checkpoints**: In regelmäßigen Intervallen sollten die Modellparameter in einem Checkpoint gespeichert werden (siehe
dazu die [TensorFlow Dokumentation](https://www.tensorflow.org/guide/saved_model)). Die Routine sollte zudem in der Lage
sein das Training eines Modells bei einem gegebenen Checkpoint wiederaufzunehmen.
* **Lernrate**: Das Aufgabenblatt gibt vor, dass beim Start optional auswählbar sein soll, ob die Lernrate beim Stagnieren
der Modellperformance während des Trainings halbiert werden soll.

#### Entscheidungsfragen

* Welchen Optimizer sollen wir verwenden? (In der Vorlesung meinten die 
[Adam](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) wäre für MT Standard)
* Nach welchem Kriterium ist das Stagnieren der Modellperformance zu erkennen? (Bezieht sich auf Anpassung der Lernrate)
* Sollen wir weitere Strategien zur Anpassung der Lernrate einführen? In [BL17] sind davon ein paar gelistet, so z.B. 
die Momentum-Basierte Optimierung ([BL17, S. 74-77]).