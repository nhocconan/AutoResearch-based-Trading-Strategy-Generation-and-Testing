#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R1_S1_Wick_Reversal"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily range for Camarilla levels (previous day's range)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1 and S1 levels
    r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to current timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Weekly trend filter: EMA34 on weekly close
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 for volume MA and Camarilla calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema_1w = ema_34_1w_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Enter long: Price touches or goes below S1 and closes back above it (bullish reversal)
            # Only in weekly uptrend (price above weekly EMA34)
            if low[i] <= s1_level and close[i] > s1_level and close[i] > ema_1w and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: Price touches or goes above R1 and closes back below it (bearish reversal)
            # Only in weekly downtrend (price below weekly EMA34)
            elif high[i] >= r1_level and close[i] < r1_level and close[i] < ema_1w and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price touches or goes above R1 (failed bullish reversal)
            if high[i] >= r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price touches or goes below S1 (failed bearish reversal)
            if low[i] <= s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals