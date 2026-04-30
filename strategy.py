#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation
# Donchian channels provide robust price structure; 1d EMA50 filters for higher-timeframe trend alignment.
# Volume spike (1.8x 20-period average) confirms institutional participation.
# Discrete sizing 0.25 to balance return and drawdown. Target: 100-180 total trades over 4 years (25-45/year).
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.

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
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        # Donchian(20) breakout levels using prior 20 bars (excluding current)
        lookback_start = max(0, i-20)
        lookback_end = i  # exclude current bar
        if lookback_end - lookback_start < 20:
            # Not enough lookback, hold current signal
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
            continue
            
        highest_high = np.max(high[lookback_start:lookback_end])
        lowest_low = np.min(low[lookback_start:lookback_end])
        
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above Donchian upper AND above 1d EMA50 (uptrend)
                if curr_close > highest_high and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian lower AND below 1d EMA50 (downtrend)
                elif curr_close < lowest_low and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below Donchian lower or below 1d EMA50
            if curr_close < lowest_low or curr_close < curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian upper or above 1d EMA50
            if curr_close > highest_high or curr_close > curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals