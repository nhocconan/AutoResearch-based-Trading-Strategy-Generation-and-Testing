#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Donchian channels for breakout signals ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_period = 20
    upper_donchian = pd.Series(high_1d).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_donchian = pd.Series(low_1d).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === Daily ATR for volatility filter ===
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align indicators to 12h timeframe
    upper_donchian_aligned = align_htf_to_ltf(prices, df_1d, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_1d, lower_donchian)
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    
    # === 12h volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(upper_donchian_aligned[i]) or 
            np.isnan(lower_donchian_aligned[i]) or 
            np.isnan(atr_10_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        upper_dc = upper_donchian_aligned[i]
        lower_dc = lower_donchian_aligned[i]
        atr_val = atr_10_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long on Donchian upper breakout with volatility filter and volume
            if (price_close > upper_dc and 
                atr_val > 0 and  # Ensure ATR is valid
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian lower breakdown with volatility filter and volume
            elif (price_close < lower_dc and 
                  atr_val > 0 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses mid-point of Donchian channel or volatility drops
            mid_point = (upper_dc + lower_dc) / 2
            if position == 1 and (price_close < mid_point or vol_ratio_val < 0.8):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > mid_point or vol_ratio_val < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_ATR_Volume_Filter"
timeframe = "12h"
leverage = 1.0