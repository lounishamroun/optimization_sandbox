
def reverse_engineer_debug_(target,instance_of=None,print_dir=False):
  print(target)
  try:
    print(f'{getattr(target,"shape")}')
  except:
    AttributeError("No attribute 'Shape'")
  print(f'Type:{type(target)}') 
  if instance_of is not None:
    if isinstance(target,instance_of)==True:
      print(f"Target is an instance of {instance_of}")
    else:
      print(f"Target isn't an instance of {instance_of}")
  if print_dir==True:
    print([attr for attr in dir(target) if not attr.startswith("_")])