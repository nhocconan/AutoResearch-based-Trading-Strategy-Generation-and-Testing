#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_R1_S1_Breakout_Volume_Trend_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Camarilla Pivot Levels (based on previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and ranges from previous day's data
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R1 = close_1d + (range_1d * 1.1 / 12)
    S1 = close_1d - (range_1d * 1.1 / 12)
    R2 = close_1d + (range_1d * 1.1 / 6)
    S2 = close_1d - (range_1d * 1.1 / 6)
    
    # Align to 4h timeframe (use previous day's levels)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # === 4h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average) with min_periods
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 4h: EMA 34 for trend filter ===
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
        R2_val = R2_aligned[i]
        S2_val = S2_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema34_val) or np.isnan(vol_ratio_val) or 
            np.isnan(R1_val) or np.isnan(S1_val) or 
            np.isnan(R2_val) or np.isnan(S2_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and uptrend
            if close_val > R1_val and vol_ratio_val > 2.2 and close_val > ema34_val:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below S1 with volume confirmation and downtrend
            elif close_val < S1_val and vol_ratio_val > 2.2 and close_val < ema34_val:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: Price falls below S1 or trend turns down
            if close_val < S1_val or close_val < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: Price rises above R1 or trend turns up
            if close_val > R1_val or close_val > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals