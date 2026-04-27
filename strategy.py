#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d trend filter and volume confirmation.
# Uses 1d EMA50 for trend direction and volume spike for confirmation.
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 20-50 trades/year to avoid excessive fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on daily close with proper min_periods
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channel (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = 20  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily EMA50 (using previous bar)
        if i > 0 and not np.isnan(ema_50_1d_aligned[i-1]):
            trend_up = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            trend_down = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: price breaks above Donchian upper + uptrend + volume spike
            if (high[i] > high_max_20[i-1] and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + downtrend + volume spike
            elif (low[i] < low_min_20[i-1] and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian lower or trend turns down
            if (low[i] < low_min_20[i-1] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper or trend turns up
            if (high[i] > high_max_20[i-1] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_1dEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0