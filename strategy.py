#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume spike confirmation.
Goes long when price breaks above R3 with 1d uptrend and volume > 1.5x 20-period average,
short when price breaks below S3 with 1d downtrend and volume > 1.5x 20-period average.
Uses discrete sizing (0.25) to minimize fees. Target: 15-30 trades/year.
R3/S3 are stronger reversal levels than R1/S1, reducing false breakouts and overtrading.
Works in bull via breakouts with trend, in bear via mean reversion at extremes.
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
    
    # Get 1d data for Camarilla calculations (using daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today using yesterday's OHLC
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])  # yesterday's close
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])   # yesterday's high
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])     # yesterday's low
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 0.55 * camarilla_range  # R3 level
    s3 = prev_close - 0.55 * camarilla_range  # S3 level
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 1d data for trend filter (using daily close)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R3, 1d uptrend (close > EMA50), volume spike
            long_signal = (close[i] > r3_aligned[i]) and (close[i] > ema_50_1d_aligned[i]) and vol_spike[i]
            # Short: price breaks below S3, 1d downtrend (close < EMA50), volume spike
            short_signal = (close[i] < s3_aligned[i]) and (close[i] < ema_50_1d_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price closes below S3 (mean reversion) or 1d trend turns down
            exit_signal = (close[i] < s3_aligned[i]) or (close[i] < ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price closes above R3 (mean reversion) or 1d trend turns up
            exit_signal = (close[i] > r3_aligned[i]) or (close[i] > ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0