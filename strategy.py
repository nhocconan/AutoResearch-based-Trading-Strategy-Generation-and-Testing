#!/usr/bin/env python3
"""
4h Camarilla R3S3 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels act as strong support/resistance. 
Breakouts above R3 or below S3 with 1d EMA34 trend alignment and volume spike 
signal strong momentum moves. Works in both bull (breakouts up in uptrend) 
and bear (breakouts down in downtrend) markets. Targets 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (need 1d OHLC)
    # We'll calculate daily Camarilla and align to 4h
    df_1d_copy = df_1d.copy()
    # Typical Camarilla calculation: based on previous day's range
    close_1d = df_1d_copy['close'].values
    high_1d = df_1d_copy['high'].values
    low_1d = df_1d_copy['low'].values
    
    # Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    # PP = (high + low + close) / 3
    # Range = high - low
    # R3 = close + Range * 1.1 / 2
    # S3 = close - Range * 1.1 / 2
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3 = close_1d + range_1d * 1.1 / 2.0
    s3 = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=1)
    
    # Calculate 20-period volume MA for 4h volume confirmation
    vol_ma_20_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        vol_ma_4h = vol_ma_20_4h[i]
        
        # Volume confirmation: current 4h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_4h
        
        if position == 0:
            # Look for entry signals
            # Long: Break above R3 AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_close > r3_level and 
                         curr_close > ema_trend and volume_confirm)
            # Short: Break below S3 AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_close < s3_level and 
                          curr_close < ema_trend and volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Price falls back below R3 OR falls below EMA34
            if (curr_close < r3_level or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Price rises back above S3 OR rises above EMA34
            if (curr_close > s3_level or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0