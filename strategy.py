#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout + 1d EMA34 trend filter + volume confirmation
# Camarilla levels provide precise intraday support/resistance; 1d EMA34 filters for daily trend alignment.
# Volume spike (2.0x 20-period average) confirms institutional participation.
# Uses 1h timeframe for entry timing, 4h/1d for signal direction. Discrete sizing 0.20 to minimize fee churn.
# Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R3, S3, R4, S4) using prior 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels based on prior 4h bar (OHLC)
    camarilla_high = np.maximum(high_4h, close_4h)
    camarilla_low = np.minimum(low_4h, close_4h)
    camarilla_range = camarilla_high - camarilla_low
    
    # Avoid division by zero
    camarilla_range = np.where(camarilla_range == 0, 1e-10, camarilla_range)
    
    # Calculate Camarilla levels for prior 4h bar
    camarilla_r3 = camarilla_high + 1.1 * camarilla_range * 1.25 / 4
    camarilla_s3 = camarilla_low - 1.1 * camarilla_range * 1.25 / 4
    camarilla_r4 = camarilla_high + 1.1 * camarilla_range * 1.5 / 2
    camarilla_s4 = camarilla_low - 1.1 * camarilla_range * 1.5 / 2
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_r4 = camarilla_r4_aligned[i]
        curr_s4 = camarilla_s4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above Camarilla R3 AND above 1d EMA34 (uptrend)
                if curr_close > curr_r3 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.20
                    position = 1
                # Bearish entry: price breaks below Camarilla S3 AND below 1d EMA34 (downtrend)
                elif curr_close < curr_s3 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below Camarilla S3 or below 1d EMA34
            if curr_close < curr_s3 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price rises above Camarilla R3 or above 1d EMA34
            if curr_close > curr_r3 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals