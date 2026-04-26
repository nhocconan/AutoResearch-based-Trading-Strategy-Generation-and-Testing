#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla R3 level AND 1d trend is up (close > EMA50) AND volume > 2.0x 20-period average volume. Enter short when price breaks below Camarilla S3 level AND 1d trend is down (close < EMA50) AND volume > 2.0x 20-period average volume. Uses discrete sizing (0.0, ±0.25) to limit fee drag. Camarilla R3/S3 levels provide stronger breakout confirmation than R1/S1, reducing false breakouts. Volume spike filter ensures participation. 1d EMA50 trend filter ensures alignment with higher timeframe momentum. Designed to generate ~12-30 trades per year on BTC/ETH/SOL with Sharpe > 0 in both bull and bear regimes.
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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # need at least previous bar for Camarilla and EMA
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_raw = df_1d['close'].values  # raw 1d close for Camarilla calculation
    
    # Camarilla levels: based on previous day's range
    # R3 = close + 1.1*(high - low)/4
    # S3 = close - 1.1*(high - low)/4
    # Using previous completed 1d bar to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d_raw, 1)
    
    # First bar has no previous bar, set to NaN
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + 1.1 * camarilla_range / 4
    s3 = prev_close_1d - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.0x 20-period average (balanced for trade frequency)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup and volume MA warmup
    start_idx = max(50, 20)  # EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
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
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: breakout above R3 + volume spike + 1d uptrend
            long_signal = breakout_up and volume_spike[i] and trend_uptrend
            
            # Short: breakout below S3 + volume spike + 1d downtrend
            short_signal = breakout_down and volume_spike[i] and trend_downtrend
            
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
            # Exit: price falls below R3 OR trend change to downtrend
            if close[i] < r3_aligned[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above S3 OR trend change to uptrend
            if close[i] > s3_aligned[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0