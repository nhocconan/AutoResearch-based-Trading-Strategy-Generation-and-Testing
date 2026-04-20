#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Camarilla_R1S1_Breakout_Volume_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 12h: Calculate 12h EMA for trend filter ===
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === 1d: Calculate Camarilla pivot levels (using previous day's data) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC for today's levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Set first day's values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels: R1, S1
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = close[i]
        ema_val = ema_12h_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(r1_level) or np.isnan(s1_level) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation AND above 12h EMA
            if (close_val > r1_level and   # Break above R1
                vol_ratio_val > 1.8 and    # Volume confirmation
                close_val > ema_val):      # Above 12h EMA (uptrend)
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation AND below 12h EMA
            elif (close_val < s1_level and   # Break below S1
                  vol_ratio_val > 1.8 and    # Volume confirmation
                  close_val < ema_val):      # Below 12h EMA (downtrend)
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below R1 (reversion to mean) OR below 12h EMA
            if close_val < r1_level or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above S1 (reversion to mean) OR above 12h EMA
            if close_val > s1_level or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals