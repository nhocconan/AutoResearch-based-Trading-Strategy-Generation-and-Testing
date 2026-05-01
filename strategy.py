#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Uses 4h EMA50 for trend direction (bull/bear) and 1h Camarilla R3/S3 for precise entry timing.
# Volume spike (>1.5x 20-bar average) confirms breakout strength.
# Session filter (08-20 UTC) reduces noise trades. Discrete sizing 0.20 manages drawdown.
# Target: 15-35 trades/year (60-140 over 4 years) to avoid fee drag.
# Works in bull (breakouts with trend) and bear (faded breakouts against trend) by aligning with 4h structure.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend direction
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla pivot points (using prior bar OHLC)
    # We'll calculate pivot from prior bar's OHLC for current bar's bias
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r3 = prev_close + (range_val * 1.1 / 4)
    s3 = prev_close - (range_val * 1.1 / 4)
    
    # Volume spike: >1.5x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(pivot[i]) or
            np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_pivot = pivot[i]
        curr_r3 = r3[i]
        curr_s3 = s3[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Trend direction from 4h EMA50
        uptrend = curr_close > curr_ema_50_4h
        downtrend = curr_close < curr_ema_50_4h
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Uptrend AND price breaks above R3 AND volume spike
            if (uptrend and 
                curr_close > curr_r3 and 
                curr_volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: Downtrend AND price breaks below S3 AND volume spike
            elif (downtrend and 
                  curr_close < curr_s3 and 
                  curr_volume_spike):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Downtrend OR price breaks below pivot
            if (not uptrend or 
                curr_close < curr_pivot):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Uptrend OR price breaks above pivot
            if (not downtrend or 
                curr_close > curr_pivot):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals