#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_v1
Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla R3 level from previous 1d bar AND 1d trend is up (close > EMA34) AND volume > 1.5x 20-period average volume. Enter short when price breaks below Camarilla S3 level AND 1d trend is down (close < EMA34) AND volume > 1.5x 20-period average volume. Exit on trend reversal or price retracing back inside the Camarilla H3-L3 range. Uses discrete sizing (0.0, ±0.25) to limit fee drag. Designed for 12-25 trades per year per symbol with Sharpe > 0 in both bull and bear regimes by avoiding overtrading and focusing on high-confluence breakouts aligned with higher timeframe trend.
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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_raw = df_1d['close'].values
    
    # Previous completed 1d bar (shift by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d_raw, 1)
    
    # First bar has no previous day
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + 1.1 * camarilla_range / 2
    s3 = prev_close_1d - 1.1 * camarilla_range / 2
    r4 = prev_close_1d + 1.25 * camarilla_range / 2
    s4 = prev_close_1d - 1.25 * camarilla_range / 2
    r3_s4_avg = (r3 + s4) / 2  # midpoint for exit
    s3_r4_avg = (s3 + r4) / 2  # midpoint for exit
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_s4_avg_aligned = align_htf_to_ltf(prices, df_1d, r3_s4_avg)
    s3_r4_avg_aligned = align_htf_to_ltf(prices, df_1d, s3_r4_avg)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup and volume MA warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            np.isnan(r3_s4_avg_aligned[i]) or
            np.isnan(s3_r4_avg_aligned[i])):
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
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
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
            # Exit: trend change to downtrend OR price retreats below midpoint (R3/S4)
            if not trend_uptrend or close[i] < r3_s4_avg_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend change to uptrend OR price rises above midpoint (S3/R4)
            if not trend_downtrend or close[i] > s3_r4_avg_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_v1"
timeframe = "12h"
leverage = 1.0