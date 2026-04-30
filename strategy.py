#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla R3/S3 breakout + 1d EMA50 trend filter + volume confirmation
# Weekly Camarilla levels provide key weekly support/resistance; 1d EMA50 filters for daily trend alignment.
# Volume spike (2.0x 20-period average) confirms institutional participation.
# Uses 6h timeframe for entry timing, 1w/1d for signal direction. Discrete sizing 0.25 to balance return and drawdown.
# Target: 80-120 total trades over 4 years (20-30/year) to avoid fee drag.

name = "6h_Weekly_Camarilla_R3S3_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # Load weekly data ONCE before loop for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (R3, S3, R4, S4) using prior weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels based on prior weekly bar (OHLC)
    camarilla_high = np.maximum(high_1w, close_1w)
    camarilla_low = np.minimum(low_1w, close_1w)
    camarilla_range = camarilla_high - camarilla_low
    
    # Avoid division by zero
    camarilla_range = np.where(camarilla_range == 0, 1e-10, camarilla_range)
    
    # Calculate Camarilla levels for prior weekly bar
    camarilla_r3 = camarilla_high + 1.1 * camarilla_range * 1.25 / 4
    camarilla_s3 = camarilla_low - 1.1 * camarilla_range * 1.25 / 4
    camarilla_r4 = camarilla_high + 1.1 * camarilla_range * 1.5 / 2
    camarilla_s4 = camarilla_low - 1.1 * camarilla_range * 1.5 / 2
    
    # Align Camarilla levels to 6h timeframe (wait for completed weekly bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Load daily data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_r4 = camarilla_r4_aligned[i]
        curr_s4 = camarilla_s4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above Camarilla R3 AND above 1d EMA50 (uptrend)
                if curr_close > curr_r3 and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Camarilla S3 AND below 1d EMA50 (downtrend)
                elif curr_close < curr_s3 and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below Camarilla S3 or below 1d EMA50
            if curr_close < curr_s3 or curr_close < curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Camarilla R3 or above 1d EMA50
            if curr_close > curr_r3 or curr_close > curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals