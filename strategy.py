#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm
Hypothesis: On daily timeframe, Camarilla R3/S3 breakouts combined with 1-week EMA trend filter and volume confirmation.
Weekly EMA ensures alignment with major trend, reducing false breakouts in choppy markets. Camarilla R3/S3 levels provide
strong intraday support/resistance. Volume spike confirms institutional participation. Designed for 7-25 trades/year
(30-100 over 4 years) with discrete position sizing to minimize fee drag. Works in bull markets (breakout continuation)
and bear markets (fade at extreme levels) via confluence filtering.
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
    
    # 1d data for Camarilla calculation (based on prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Prior day OHLC for Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    daily_range = prev_high - prev_low
    # Camarilla R3 and S3 levels
    r3 = prev_close + daily_range * 1.1000 / 4  # R3 = C + (H-L)*1.1/4
    s3 = prev_close - daily_range * 1.1000 / 4  # S3 = C - (H-L)*1.1/4
    
    # Align Camarilla levels to 1d timeframe (completed daily bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2.0 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for prior day data (1) + EMA (34) + volume MA (20)
    start_idx = max(35, 20)  # 35 for prior day + EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla R3/S3 breakout + volume spike + 1w EMA trend alignment
            long_breakout = curr_high > r3_aligned[i]
            short_breakout = curr_low < s3_aligned[i]
            
            # Trend filter: price must be on correct side of 1w EMA
            long_trend = curr_close > ema_34_1w_aligned[i]
            short_trend = curr_close < ema_34_1w_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and long_trend)
            short_entry = (short_breakout and volume_spike[i] and short_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Camarilla R3 (failed breakout) or trend reverses
            if curr_close < r3_aligned[i] or curr_close < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Camarilla S3 (failed breakout) or trend reverses
            if curr_close > s3_aligned[i] or curr_close > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0