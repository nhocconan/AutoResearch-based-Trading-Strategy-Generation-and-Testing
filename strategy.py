#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_WeeklyTrend_VolumeConfirm
Hypothesis: 6h Camarilla R3/S3 breakout with 1w trend filter (price > 1w EMA50) and 1d volume spike confirmation (>1.5x 20-period average). Designed for 12-30 trades/year. Works in bull markets via breakouts above R3 with weekly uptrend, and in bear markets via breakdowns below S3 with weekly downtrend. Uses discrete sizing (0.25) to minimize fees.
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
    
    # Get 1d data for Camarilla pivot calculation (based on prior 1d candle)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar: based on prior day's OHLC
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using prior 1d candle to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use prior day's OHLC for today's levels (no look-ahead)
    prior_high_1d = np.roll(high_1d, 1)
    prior_low_1d = np.roll(low_1d, 1)
    prior_close_1d = np.roll(close_1d, 1)
    prior_high_1d[0] = np.nan
    prior_low_1d[0] = np.nan
    prior_close_1d[0] = np.nan
    
    camarilla_r3_1d = prior_close_1d + 1.1 * (prior_high_1d - prior_low_1d) / 2
    camarilla_s3_1d = prior_close_1d - 1.1 * (prior_high_1d - prior_low_1d) / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # 1w trend filter: EMA50 on 1w close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d volume spike: >1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.5 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: break above R3 + weekly uptrend + volume spike
            long_signal = (close[i] > r3_aligned[i]) and (close[i] > ema_50_1w_aligned[i]) and (vol_spike_aligned[i] > 0.5)
            # Short: break below S3 + weekly downtrend + volume spike
            short_signal = (close[i] < s3_aligned[i]) and (close[i] < ema_50_1w_aligned[i]) and (vol_spike_aligned[i] > 0.5)
            
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
            # Exit when price closes below R3 (failed breakout) or below weekly EMA (trend change)
            exit_signal = (close[i] < r3_aligned[i]) or (close[i] < ema_50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price closes above S3 (failed breakdown) or above weekly EMA (trend change)
            exit_signal = (close[i] > s3_aligned[i]) or (close[i] > ema_50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_WeeklyTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0