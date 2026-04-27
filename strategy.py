#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_WeeklyTrend_VolumeSpike
Hypothesis: Daily Camarilla R3/S3 breakout with weekly EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND price > weekly EMA50 AND volume spike.
Short when price breaks below Camarilla S3 AND price < weekly EMA50 AND volume spike.
Exit on opposite Camarilla level break or loss of weekly EMA50 alignment.
Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year) with low fee drag.
Weekly trend filter ensures alignment with higher timeframe momentum, reducing false breakouts.
Works in bull markets (breakouts with weekly uptrend) and bear markets (breakdowns with weekly downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from prior day (1d)
    df_1d = get_htf_data(prices, '1d')
    # Prior day OHLC (shifted by 1 to avoid look-ahead)
    prev_close = pd.Series(df_1d['close'].values).shift(1)
    prev_high = pd.Series(df_1d['high'].values).shift(1)
    prev_low = pd.Series(df_1d['low'].values).shift(1)
    
    # Camarilla levels
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 1d (no shift needed as they're for current day)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Weekly EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for 1d Camarilla (2d), weekly EMA50 (~50 weeks), volume avg
    start_idx = max(48, 100, 20)  # 2 days for prior data, ~50 weeks for EMA50, 20 for vol avg
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: Camarilla breakout with weekly EMA50 alignment and volume spike
            # Long: Close > Camarilla R3 AND price > weekly EMA50 AND volume spike
            # Short: Close < Camarilla S3 AND price < weekly EMA50 AND volume spike
            long_condition = (close_val > r3_val and 
                            close_val > ema_val and 
                            vol_spike)
            short_condition = (close_val < s3_val and 
                             close_val < ema_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Camarilla S3 OR loses weekly EMA50 alignment
            if close_val < s3_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Camarilla R3 OR loses weekly EMA50 alignment
            if close_val > r3_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0