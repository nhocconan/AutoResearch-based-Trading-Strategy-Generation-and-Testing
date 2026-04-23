#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation.
Long when Williams %R(14) crosses above -80 (oversold) AND 1d close > 1d EMA50 (uptrend) AND volume > 1.3x 20-period MA.
Short when Williams %R(14) crosses below -20 (overbought) AND 1d close < 1d EMA50 (downtrend) AND volume > 1.3x 20-period MA.
Exit when Williams %R crosses the opposite threshold (-20 for long, -80 for short) or 1d trend reverses.
Williams %R identifies exhaustion points in ranging/bear markets; 1d EMA50 filters counter-trend trades; volume confirms momentum.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Calculate 6h Williams %R(14)
    williams_period = 14
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, -100 * (highest_high - close) / denominator, -50)
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(williams_period, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate 1d trend: close > EMA50 = uptrend, close < EMA50 = downtrend
        # Need 1d close price aligned to 6h bars
        df_1d = get_htf_data(prices, '1d')  # Reload for close (could optimize but safe)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        trend_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume filter: 6h volume > 1.3x 20-period MA
        vol_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND uptrend AND volume filter
            if i > start_idx and williams_r[i-1] <= -80 and williams_r[i] > -80 and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND downtrend AND volume filter
            elif i > start_idx and williams_r[i-1] >= -20 and williams_r[i] < -20 and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -20 (overbought) OR 1d trend turns down
                if williams_r[i] >= -20 or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -80 (oversold) OR 1d trend turns up
                if williams_r[i] <= -80 or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_MeanReversion_1dEMA50_Trend_VolumeFilter"
timeframe = "6h"
leverage = 1.0