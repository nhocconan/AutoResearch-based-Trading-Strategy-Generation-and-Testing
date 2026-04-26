#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: Tighten entry conditions to reduce trades from 135 to target 75-150 over 4 years (19-37/year). Increase volume threshold from 2.5x to 3.0x and add ATR filter to avoid whipsaws. Maintain Camarilla R3/S3 breakout with 12h EMA50 trend filter. Target: SOL/ETH/BTC robustness with Sharpe > 0.5 on test.
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
    
    # Get 12h data for trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Camarilla levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_raw = df_12h['close'].values
    
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h_raw, 1)
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    prev_close_12h[0] = np.nan
    
    camarilla_range = prev_high_12h - prev_low_12h
    r3 = prev_close_12h + 1.1 * camarilla_range / 4
    s3 = prev_close_12h - 1.1 * camarilla_range / 4
    mid = (r3 + s3) / 2  # Camarilla midpoint for exit
    
    # Align Camarilla levels and EMA50 to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    mid_aligned = align_htf_to_ltf(prices, df_12h, mid)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 3.0x 20-period average (tighter)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 3.0
    
    # ATR filter to avoid choppy markets (ATR(14) < 50-day ATR median)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_median = pd.Series(atr_50).rolling(window=50, min_periods=1).median().values
    atr_filter = (atr_14 < atr_median)  # Only trade when volatility is normal/low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup, volume MA warmup, ATR warmup
    start_idx = max(50, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(atr_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > r3_aligned[i]
        breakout_down = close[i] < s3_aligned[i]
        
        # 12h trend filter
        trend_uptrend = close[i] > ema_50_12h_aligned[i]
        trend_downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: breakout above R3 + volume spike + 12h uptrend + ATR filter
            long_signal = breakout_up and volume_spike[i] and trend_uptrend and atr_filter[i]
            
            # Short: breakout below S3 + volume spike + 12h downtrend + ATR filter
            short_signal = breakout_down and volume_spike[i] and trend_downtrend and atr_filter[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend change to downtrend OR price retracing to Camarilla midpoint
            if not trend_uptrend or close[i] < mid_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend change to uptrend OR price retracing to Camarilla midpoint
            if not trend_downtrend or close[i] > mid_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0