#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Camarilla_R1S1_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Weekly Camarilla Pivot Levels (based on previous week) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot and ranges from previous week's data
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    R1 = close_1w + (range_1w * 1.1 / 12)
    S1 = close_1w - (range_1w * 1.1 / 12)
    
    # Align to 12h timeframe (use previous week's levels)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # === 12h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average) with min_periods
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 12h: EMA 34 for trend filter ===
    close_series = pd.Series(close)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        ema34_val = ema34[i]
        vol_ratio_val = vol_ratio[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema34_val) or np.isnan(vol_ratio_val) or 
            np.isnan(R1_val) or np.isnan(S1_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and uptrend
            if close_val > R1_val and vol_ratio_val > 1.8 and close_val > ema34_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and downtrend
            elif close_val < S1_val and vol_ratio_val > 1.8 and close_val < ema34_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls below S1 or trend turns down
            if close_val < S1_val or close_val < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above R1 or trend turns up
            if close_val > R1_val or close_val > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals