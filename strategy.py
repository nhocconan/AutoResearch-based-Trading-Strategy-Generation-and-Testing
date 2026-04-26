#!/usr/bin/env python3
"""
6h_ADX_Di_StochOsc_1dTrend_VolumeConfirm_v1
Hypothesis: Combine ADX DI crossover (trend strength/direction) with Stochastic oscillator for timing, filtered by 1d EMA50 trend and volume confirmation. 
Long when +DI > -DI (uptrend) + Stoch %K crosses above %D from below 20 (oversold bounce) + price > 1d EMA50 + volume > 1.5x average.
Short when -DI > +DI (downtrend) + Stoch %K crosses below %D from above 80 (overbought rejection) + price < 1d EMA50 + volume > 1.5x average.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).
Works in bull markets via trend-following entries and in bear markets via overbought/oversold mean-reversion within the trend filter.
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
    
    # Calculate ADX and DI (14-period)
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high[1:] - high[:-1]])
    down_move = np.concatenate([[np.nan], low[:-1] - low[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (using EMA-like smoothing with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    def smooth_series(data):
        result = np.full_like(data, np.nan)
        for i in range(len(data)):
            if np.isnan(data[i]):
                if i == 0:
                    result[i] = np.nan
                else:
                    result[i] = result[i-1]
            else:
                if i == 0 or np.isnan(result[i-1]):
                    result[i] = data[i]
                else:
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smoothed = smooth_series(tr)
    plus_dm_smoothed = smooth_series(plus_dm)
    minus_dm_smoothed = smooth_series(minus_dm)
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_smoothed / np.where(tr_smoothed == 0, np.nan, tr_smoothed)
    minus_di = 100 * minus_dm_smoothed / np.where(tr_smoothed == 0, np.nan, tr_smoothed)
    
    # Calculate Stochastic Oscillator (14,3,3)
    lookback = 14
    lowest_low = np.full_like(low, np.nan)
    highest_high = np.full_like(high, np.nan)
    
    for i in range(lookback-1, len(low)):
        window_low = low[i-lookback+1:i+1]
        window_high = high[i-lookback+1:i+1]
        lowest_low[i] = np.nanmin(window_low)
        highest_high[i] = np.nanmax(window_high)
    
    stoch_k = 100 * (close - lowest_low) / np.where((highest_high - lowest_low) == 0, np.nan, (highest_high - lowest_low))
    
    # Smooth %K to get %D (3-period SMA of %K)
    stoch_d = np.full_like(stoch_k, np.nan)
    for i in range(2, len(stoch_k)):
        if not (np.isnan(stoch_k[i-2]) or np.isnan(stoch_k[i-1]) or np.isnan(stoch_k[i])):
            stoch_d[i] = np.nanmean(stoch_k[i-2:i+1])
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(14, 20, 50) + 2  # +2 for Stoch %D
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or np.isnan(stoch_k[i]) or 
            np.isnan(stoch_d[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: +DI > -DI (uptrend) + Stoch %K crosses above %D from below 20 + price > 1d EMA50 + volume spike
        if (plus_di[i] > minus_di[i] and 
            stoch_k[i-1] <= stoch_d[i-1] and stoch_k[i] > stoch_d[i] and stoch_k[i] < 20 and
            close[i] > ema_50_1d_aligned[i] and volume_spike[i]):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: -DI > +DI (downtrend) + Stoch %K crosses below %D from above 80 + price < 1d EMA50 + volume spike
        elif (minus_di[i] > plus_di[i] and 
              stoch_k[i-1] >= stoch_d[i-1] and stoch_k[i] < stoch_d[i] and stoch_k[i] > 80 and
              close[i] < ema_50_1d_aligned[i] and volume_spike[i]):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: trend reversal or loss of momentum
        elif position == 1 and (plus_di[i] <= minus_di[i] or stoch_k[i] > 80):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (minus_di[i] <= plus_di[i] or stoch_k[i] < 20):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_ADX_Di_StochOsc_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0