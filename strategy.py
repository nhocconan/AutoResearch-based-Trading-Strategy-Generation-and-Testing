#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h trend filter (EMA34) and volume confirmation
# Works in bull markets via breakout momentum, in bear markets via trend filter avoiding false breakouts
# Target: 20-40 trades/year (~80-160 total over 4 years) to minimize fee drag
name = "4h_12h_Donchian20_EMA34_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:
        return np.zeros(n)
    
    # === 12h EMA34 for trend filter ===
    close_12h = df_12h['close'].values
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # === 4h Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma20 > 0, volume / vol_ma20, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is invalid
        if (np.isnan(ema_34_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = prices['close'].iloc[i]
        ema_val = ema_34_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        vol = vol_ratio[i]
        
        if position == 0:
            # Long: Break above Donchian high with volume, in uptrend (price > EMA34)
            if close_val > upper and vol > 1.5 and close_val > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume, in downtrend (price < EMA34)
            elif close_val < lower and vol > 1.5 and close_val < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below Donchian low OR trend changes
            if close_val < lower or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above Donchian high OR trend changes
            if close_val > upper or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals