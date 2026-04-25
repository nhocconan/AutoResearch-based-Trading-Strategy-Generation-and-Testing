#!/usr/bin/env python3
"""
1h_VolumeSpike_4hCamarillaBreakout_1dTrend
Hypothesis: On 1h timeframe, enter long when price breaks above 4h Camarilla R3 with volume spike (>2x 20-bar mean) and 1d uptrend (close > 1d EMA50). Enter short when price breaks below 4h Camarilla S3 with volume spike and 1d downtrend (close < 1d EMA50). Use session filter (08-20 UTC) to avoid low-liquidity hours. Discrete position size 0.20 to minimize fee churn. Designed for 15-35 trades/year per symbol, effective in bull markets via breakouts and bear markets via trend-following shorts.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Camarilla levels from previous 4h bar
    camarilla_r3 = close_4h + 1.1 * (high_4h - low_4h)
    camarilla_s3 = close_4h - 1.1 * (high_4h - low_4h)
    
    # Align Camarilla levels to 1h timeframe (use previous bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside session or data not ready
        if not in_session[i] or \
           np.isnan(camarilla_r3_aligned[i]) or \
           np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(vol_mean_20[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 in uptrend (close > 1d EMA50) with volume confirmation
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema_50_1d_aligned[i]) and vol_confirm[i]
            # Short: price breaks below Camarilla S3 in downtrend (close < 1d EMA50) with volume confirmation
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema_50_1d_aligned[i]) and vol_confirm[i]
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit when price moves back below 1d EMA50 (trend reversal)
            exit_signal = close[i] < ema_50_1d_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price moves back above 1d EMA50 (trend reversal)
            exit_signal = close[i] > ema_50_1d_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_VolumeSpike_4hCamarillaBreakout_1dTrend"
timeframe = "1h"
leverage = 1.0