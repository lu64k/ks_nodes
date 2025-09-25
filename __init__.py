from .ks_node import KS_Load_Images_From_Folder
from .KS_text_tools import KSLoadText, KS_Save_Text, KS_Text_String, KS_Random_File_Name, KS_get_time_int
from .ks_json_tools import KS_Json_Float_Range_Filter, KS_Json_Array_Constrains_Filter, KS_Json_Key_Replace_3ways, KS_Json_Value_Eliminator, KS_Json_Extract_Key_And_Value_3ways, KS_Json_Key_Random_3ways,  KS_Json_Count, KS_JsonToString, KS_Json_loader, KS_JsonKeyReplacer, KS_JsonKeyExtractor, KS_merge_json_node, KS_make_json_node, KS_JsonlFolderMatchReader, KS_image_metadata_node, KS_Save_JSON #KS_Word_Frequency_Statistics,
from .ks_api_tools import *
NODE_CLASS_MAPPINGS = {
    "KS Text_String": KS_Text_String,
    "KS Random File Name": KS_Random_File_Name,
    "KS Save Text": KS_Save_Text,
    "KS load text": KSLoadText,
    "KS get time int": KS_get_time_int,
    "KS json float range filter": KS_Json_Float_Range_Filter,
    "KS json array constrains filter": KS_Json_Array_Constrains_Filter,
    "KS_Json_Key_Replace_3ways": KS_Json_Key_Replace_3ways,
    "KS json value eliminator": KS_Json_Value_Eliminator,
    "KS_Json_Extract_Key_And_Value_3ways": KS_Json_Extract_Key_And_Value_3ways,
    "KS_Json_Key_Random_3ways": KS_Json_Key_Random_3ways,
    "KS Json Count": KS_Json_Count,
    #"KS Word Frequency Statistics": KS_Word_Frequency_Statistics,
    "KS Load Images From Folder": KS_Load_Images_From_Folder,
    "KS Json To String": KS_JsonToString,
    "KS_Json_loader":KS_Json_loader,
    "KS JsonKeyReplacer":KS_JsonKeyReplacer,
    "KS JsonKeyExtractor":KS_JsonKeyExtractor,
    "KS_merge_json_node":KS_merge_json_node,
    "KS_make_json_node":KS_make_json_node,
    "KS_any_payload_image": KS_any_payload_image_API_Node,
    "KS JsonlFolderMatchReader": KS_JsonlFolderMatchReader,
    "KS_image_metadata_node": KS_image_metadata_node,
    "KS_Save_JSON":KS_Save_JSON

}

__all__ = ['NODE_CLASS_MAPPINGS']
