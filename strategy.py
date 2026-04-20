#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_Turtle_Channel_Breakout_With_Trend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Turtle Channel (20-period Donchian) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Upper and lower bands (20-period high/low)
    upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    
    # === 6h Trend Filter: EMA50 > EMA200 for long bias, < for short bias ===
    close_series = pd.Series(prices['close'].values)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # === Volume Filter: Current volume > 1.5x 20-period average ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema50_val = ema50[i]
        ema200_val = ema200[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(ema50_val) or np.isnan(ema200_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 12h upper channel with volume and uptrend
            if close_val > upper_val and vol_ratio_val > 1.5 and ema50_val > ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below 12h lower channel with volume and downtrend
            elif close_val < lower_val and vol_ratio_val > 1.5 and ema50_val < ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: Hold while price above lower channel and trend intact
            if close_val < lower_val or ema50_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: Hold while price below upper channel and trend intact
            if close_val > upper_val or ema50_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals