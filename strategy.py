#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrend_WeeklyVolumeSpike
Hypothesis: Trade 12h Camarilla R3/S3 breakouts with 1w EMA34 trend filter and weekly volume confirmation.
R3/S3 are stronger reversal levels reducing false breakouts. Weekly EMA34 captures major trend, weekly volume spike confirms institutional interest.
Designed for low trade frequency (12-37/year) to minimize fee drag. Works in bull/bear by following major trend with momentum entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for volatility filter on 12h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly volume spike: volume > 2.0 * 4-week average
    vol_ma_4w = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma_4w)
    
    # Calculate Camarilla levels from previous 1w bar
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1w['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1w['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1w['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 4.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    
    # Align Camarilla levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1w EMA(34), volume MA(4), ATR(14)
    start_idx = max(34, 4, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_4w[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_1w_up = close_val > ema_34_1w_aligned[i]   # 1w uptrend
        trend_1w_down = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R3 AND 1w trend up AND volume spike
            long_signal = (close_val > r3_aligned[i]) and trend_1w_up and vol_spike
            
            # Short: price breaks below S3 AND 1w trend down AND volume spike
            short_signal = (close_val < s3_aligned[i]) and trend_1w_down and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend flips down OR price closes below S3 (reversal signal)
            if (not trend_1w_up) or (close_val < s3_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price closes above R3 (reversal signal)
            if (not trend_1w_down) or (close_val > r3_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wTrend_WeeklyVolumeSpike"
timeframe = "12h"
leverage = 1.0