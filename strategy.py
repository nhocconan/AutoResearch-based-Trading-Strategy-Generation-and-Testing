#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_DynamicSize
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R3 level AND 1d trend is up (close > EMA34) AND volume > 2.5x 20-period average volume. Enter short when price breaks below Camarilla S3 level AND 1d trend is down (close < EMA34) AND volume > 2.5x 20-period average volume. Uses dynamic position sizing based on volatility (ATR) to reduce drawdowns in bear markets while maintaining Sharpe in bull markets. Target: 15-40 trades/year.
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
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_raw = df_1d['close'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d_raw, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + 1.1 * camarilla_range / 4
    s3 = prev_close_1d - 1.1 * camarilla_range / 4
    mid = (r3 + s3) / 2  # Camarilla midpoint for exit
    
    # Align Camarilla levels and EMA34 to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    mid_aligned = align_htf_to_ltf(prices, df_1d, mid)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: dynamic threshold based on volume percentile
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.maximum(volume_ma, 1e-10)
    volume_spike = volume_ratio > 2.5
    
    # ATR for dynamic position sizing (14-period)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup, ATR warmup, and volume MA warmup
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
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
                # Dynamic sizing: reduce size in high volatility (bear markets)
                vol_factor = np.clip(atr[i] / (close[i] * 0.02), 0.5, 1.0)  # Normalize ATR
                size = 0.30 * vol_factor
                signals[i] = size
                position = 1
            elif short_signal:
                # Dynamic sizing: reduce size in high volatility (bear markets)
                vol_factor = np.clip(atr[i] / (close[i] * 0.02), 0.5, 1.0)  # Normalize ATR
                size = 0.30 * vol_factor
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30 * np.clip(atr[i] / (close[i] * 0.02), 0.5, 1.0)
            # Exit: trend change to downtrend OR price retracing to Camarilla midpoint
            if not trend_uptrend or close[i] < mid_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30 * np.clip(atr[i] / (close[i] * 0.02), 0.5, 1.0)
            # Exit: trend change to uptrend OR price retracing to Camarilla midpoint
            if not trend_downtrend or close[i] > mid_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_DynamicSize"
timeframe = "4h"
leverage = 1.0