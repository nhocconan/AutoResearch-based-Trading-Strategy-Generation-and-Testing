#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily data for 12h strategy ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === ATR(14) for volatility filter ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Volume ratio (current volume / 20-day average) ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / np.where(vol_ma_20 > 0, vol_ma_20, np.nan)
    
    # === Price change since previous day open ===
    open_1d = df_1d['open'].values
    price_change = (close_1d - open_1d) / open_1d
    
    # === Align to 12h timeframe ===
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ratio_12h = align_htf_to_ltf(prices, df_1d, vol_ratio)
    price_change_12h = align_htf_to_ltf(prices, df_1d, price_change)
    
    signals = np.zeros(n)
    
    # Warmup for indicators
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(atr_12h[i]) or np.isnan(vol_ratio_12h[i]) or 
            np.isnan(price_change_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
            
        atr_val = atr_12h[i]
        vol_ratio_val = vol_ratio_12h[i]
        price_change_val = price_change_12h[i]
        
        # Exit conditions
        if position == 1:  # Long
            # Exit if volatility drops or price reverses
            if vol_ratio_val < 0.8 or price_change_val < -0.005:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short
            # Exit if volatility drops or price reverses
            if vol_ratio_val < 0.8 or price_change_val > 0.005:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions
        if position == 0:
            # Long: positive price change + volume expansion + sufficient volatility
            if (price_change_val > 0.003 and vol_ratio_val > 1.5 and atr_val > 0):
                signals[i] = 0.25
                position = 1
                continue
            # Short: negative price change + volume expansion + sufficient volatility
            elif (price_change_val < -0.003 and vol_ratio_val > 1.5 and atr_val > 0):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Volume_PriceChange_ATR_Filter"
timeframe = "12h"
leverage = 1.0