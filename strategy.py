#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly OHLC for pivot calculation ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot and R1/S1 levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1_1w = close_1w + range_1w * 1.1 / 12
    s1_1w = close_1w - range_1w * 1.1 / 12
    
    # === ATR for volatility filter (14-period) ===
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_avg = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly data to daily timeframe
    r1_daily = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_daily = align_htf_to_ltf(prices, df_1w, s1_1w)
    atr_avg_daily = align_htf_to_ltf(prices, df_1w, atr_1w_avg)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === Distance from pivot to avoid chop ===
    mid_pivot = (r1_daily + s1_daily) / 2
    dist_from_pivot = np.abs(close - mid_pivot)
    avg_dist = pd.Series(dist_from_pivot).rolling(window=50, min_periods=50).mean().values
    too_close = dist_from_pivot < (0.5 * avg_dist)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_daily[i]) or np.isnan(s1_daily[i]) or
            np.isnan(atr_avg_daily[i]) or np.isnan(volume_spike[i]) or
            np.isnan(too_close[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_level = r1_daily[i]
        s1_level = s1_daily[i]
        atr_avg = atr_avg_daily[i]
        vol_spike = volume_spike[i]
        too_close_to_pivot = too_close[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price drops below S1 or volatility drops significantly
            if price < s1_level or (i > 0 and atr_avg < atr_avg_daily[i-1] * 0.7):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above R1 or volatility drops significantly
            if price > r1_level or (i > 0 and atr_avg < atr_avg_daily[i-1] * 0.7):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume spike, sufficient volatility, and not too close to pivot
            if price > r1_level and vol_spike and atr_avg > 0 and not too_close_to_pivot:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume spike, sufficient volatility, and not too close to pivot
            elif price < s1_level and vol_spike and atr_avg > 0 and not too_close_to_pivot:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Weekly_Pivot_R1_S1_Breakout_Volume_ATRFilter_DistFilter"
timeframe = "1d"
leverage = 1.0