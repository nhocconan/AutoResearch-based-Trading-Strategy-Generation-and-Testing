#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Camarilla R3/S3 breakout with weekly trend filter and volume confirmation. 
In strong weekly trends (price above/below weekly EMA34), breakouts at R3 (long) or S3 (short) 
have higher follow-through. Volume spike confirms institutional participation. 
Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year), minimizing fee drag 
while capturing multi-day momentum in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for daily timeframe (using previous day's range)
    # Camarilla: R4 = close + 1.1*(high-low)/2, R3 = close + 1.1*(high-low)/4, 
    #            S3 = close - 1.1*(high-low)/4, S4 = close - 1.1*(high-low)/2
    # We use previous day's high/low to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    daily_range = prev_high - prev_low
    R3 = prev_close + 1.1 * daily_range / 4
    S3 = prev_close - 1.1 * daily_range / 4
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2.0 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for weekly EMA34, volume average, and Camarilla calculation
    start_idx = max(35, 20, 1)  # 35 for weekly EMA34, 20 for volume, 1 for Camarilla (needs prev day)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        weekly_trend = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: Camarilla breakout in direction of weekly trend with volume spike
            # Long: Close > R3 AND weekly trend up (close > weekly EMA34) AND volume spike
            # Short: Close < S3 AND weekly trend down (close < weekly EMA34) AND volume spike
            long_condition = close_val > R3[i] and close_val > weekly_trend and vol_spike
            short_condition = close_val < S3[i] and close_val < weekly_trend and vol_spike
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below S3 (reversion to mean) OR weekly trend turns down
            if close_val < S3[i] or close_val < weekly_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above R3 (reversion to mean) OR weekly trend turns up
            if close_val > R3[i] or close_val > weekly_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0