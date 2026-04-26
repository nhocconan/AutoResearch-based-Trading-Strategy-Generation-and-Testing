#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v2
Hypothesis: Daily Camarilla R3/S3 breakout with 1-week EMA50 trend filter and volume confirmation (2.0x). 
Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year) by using tight entry conditions: 
- Breakout must occur with 2.0x volume spike 
- Must align with 1-week EMA50 trend 
- Exit on reverse breakout or trend change 
Uses discrete position sizing (0.30) to balance return and fee drag. Works in bull/bear via 1w trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = invalid
    trend_1w = np.where(ema_50_1w_aligned > 0, 
                        np.where(close > ema_50_1w_aligned, 1, -1), 
                        0)
    
    # Calculate Camarilla pivot levels from 1d OHLC (using previous day)
    df_1d = get_htf_data(prices, '1d')
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: volume > 2.0 * volume_ma(20) for tight confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(trend_1w[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Camarilla R3/S3 breakout conditions with volume and trend confirmation
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND 1w uptrend AND volume spike (2.0x)
            if close[i] > camarilla_r3_aligned[i] and trend_1w[i] == 1 and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below Camarilla S3 AND 1w downtrend AND volume spike (2.0x)
            elif close[i] < camarilla_s3_aligned[i] and trend_1w[i] == -1 and volume_spike[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: Price falls below Camarilla S3 OR 1w trend turns down
            if close[i] < camarilla_s3_aligned[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: Price rises above Camarilla R3 OR 1w trend turns up
            if close[i] > camarilla_r3_aligned[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v2"
timeframe = "1d"
leverage = 1.0