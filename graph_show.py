res = np.load('analysis/position_sieve.npy')

import matplotlib.pyplot as plt
import numpy as np

plt.figure(dpi=200)
plt.imshow(res)
plt.ylabel('Messages ranked by frequency')
plt.xlabel('Positions')
plt.savefig('position_informative_symbols.png')

#Position informative symbols

fig,ax = plt.subplots(dpi=200)
ax.get_yaxis().set_ticks([])
plt.vlines(mean_length_evolution[-1], -1, 1, colors='red')
avg_pos = (res==0).mean(axis=0)
plt.imshow([avg_pos],cmap='gray')

plt.savefig('avg_position_informative_symbol.png')

import numpy as np
np.load('dir_save/messages/messages_300.npy')

def load_message(expe):
  """
  Load messages stored during training procedure/
  Return numpy array with all the messages
  """
  np_load_old = np.load
  messages = np.load(expe,allow_pickle=True)
  np.load = np_load_old
  return messages

# Choose epochs (between 0 and 500)
epochs=[0,200,400,600,800]
plt.figure(dpi=200)


for epoch in epochs:
  # Load messages
  messages=load_message("dir_save/messages/messages_"+str(epoch)+".npy")

  # Construct the length distribution
  length_distribution=[]
  for message in messages:
    length_distribution.append(len(message))

  # Add epoch to plot
  plt.plot(length_distribution,label="Epoch "+str(epoch))

# Plot fig
plt.title("Message length as a function of inputs ranked by frequency")
plt.xlabel("Inputs ranked by frequency")
plt.ylabel("Message length")
plt.xlim((0,100))
plt.ylim((0,32))
plt.legend()
plt.savefig('message_length_by_frequency.png')

# Get accuracy
accuracy=[]
for i in range(1,800):
  accuracy.append(np.mean(np.load("dir_save/accuracy/accuracy_"+str(i)+".npy")))

# Plot fig
fig, ax = plt.subplots(1, 1, figsize=(11,4))
ax.plot(accuracy,label="LazImpa",c="tab:blue")
ax.set_title("Accuracy evolution")
ax.set_xlabel("Training episodes")
ax.set_ylabel("Accuracy")
ax.set_xlim((0,800))
ax.set_ylim((0,1))
fig.savefig('acc_evol.png')
plt.show()

mean_length_evolution=[]

for epoch in range(800):
  # Load messages
  messages=load_message("dir_save/messages/messages_"+str(epoch)+".npy")

  # Construct the length distribution
  length_distribution=[]
  for message in messages:
    length_distribution.append(len(message))
  
  # Get the mean length
  mean_length_evolution.append(np.mean(length_distribution))

# Plot fig
fig, ax = plt.subplots(1, 1, figsize=(11,4))
ax.plot(mean_length_evolution,label="LazImpa",c="tab:blue")
ax.set_title("Mean length evolution")
ax.set_xlabel("Training episodes")
ax.set_ylabel("Accuracy")
ax.set_xlim((0,800))
ax.set_ylim((0,31))
fig.savefig('length_evol.png')
plt.show()

print("mean lenght evolution [-1]" ,mean_length_evolution[-1])