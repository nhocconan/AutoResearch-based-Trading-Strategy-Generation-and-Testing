#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian channels provide robust trend-following structure; 1d EMA50 filters for higher timeframe trend alignment.
# Volume spike (2.0x 20-period average) confirms institutional participation and reduces false breakouts.
# Discrete sizing 0.25 to balance return and risk. Target: 100-180 total trades over 4 years (25-45/year).
# Works in both bull and bear markets: trend filter adapts to 1d EMA50 direction, breakouts capture momentum in either direction.

name = "4h_Donchian20_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
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
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Calculate Donchian channels using prior 20 periods (excluding current bar)
                lookback_start = max(0, i-20)
                lookback_end = i  # exclusive
                if lookback_end - lookback_start >= 20:
                    highest_high = np.max(high[lookback_start:lookback_end])
                    lowest_low = np.min(low[lookback_start:lookback_end])
                    
                    # Bullish entry: price breaks above Donchian upper band AND above 1d EMA50 (uptrend)
                    if curr_close > highest_high and curr_close > curr_ema_50_1d:
                        signals[i] = 0.25
                        position = 1
                    # Bearish entry: price breaks below Donchian lower band AND below 1d EMA50 (downtrend)
                    elif curr_close < lowest_low and curr_close < curr_ema_50_1d:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below Donchian lower band or below 1d EMA50
            lookback_start = max(0, i-20)
            lookback_end = i  # exclusive
            if lookback_end - lookback_start >= 20:
                lowest_low = np.min(low[lookback_start:lookback_end])
                if curr_close < lowest_low or curr_close < curr_ema_50_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian upper band or above 1d EMA50
            lookback_start = max(0, i-20)
            lookback_end = i  # exclusive
            if lookback_end - lookback_start >= 20:
                highest_high = np.max(high[lookback_start:lookback_end])
                if curr_close > highest_high or curr_close > curr_ema_50_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals