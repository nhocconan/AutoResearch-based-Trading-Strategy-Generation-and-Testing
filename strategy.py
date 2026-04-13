#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1-day trend filter and volume confirmation.
Williams %R identifies overbought/oversold conditions. In trending markets (ADX > 25 on 1d),
we fade extreme readings during pullbacks: long when %R crosses above -80 from below in uptrend,
short when %R crosses below -20 from above in downtrend. Volume confirmation ensures momentum.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
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
    
    # Get 1d data for trend filter (ADX) and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day ADX for trend strength
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Trend: ADX > 25 = trending market
    trending = adx_1d > 25
    
    # Determine trend direction using +DI/-DI crossover
    # Uptrend when +DI > -DI
    uptrend = plus_di_1d > minus_di_1d
    downtrend = plus_di_1d < minus_di_1d
    
    # Calculate 1-day volume confirmation (volume > 1.3x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume_1d > (vol_ma_20 * 1.3)
    
    # Calculate Williams %R on 6h data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d indicators to 6h timeframe
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending.astype(float))
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend.astype(float))
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(trending_aligned[i]) or 
            np.isnan(uptrend_aligned[i]) or 
            np.isnan(downtrend_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Williams %R extreme + trend + volume
        # Long: %R crosses above -80 from below in uptrend with volume
        williams_r_cross_up = (williams_r[i] > -80) and (williams_r[i-1] <= -80) if i > 0 else False
        long_entry = williams_r_cross_up and trending_aligned[i] and uptrend_aligned[i] and vol_confirm_aligned[i]
        
        # Short: %R crosses below -20 from above in downtrend with volume
        williams_r_cross_down = (williams_r[i] < -20) and (williams_r[i-1] >= -20) if i > 0 else False
        short_entry = williams_r_cross_down and trending_aligned[i] and downtrend_aligned[i] and vol_confirm_aligned[i]
        
        # Exit when %R returns to opposite extreme (mean reversion within trend)
        exit_long = position == 1 and williams_r[i] > -20
        exit_short = position == -1 and williams_r[i] < -80
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_williams_r_trend_volume"
timeframe = "6h"
leverage = 1.0