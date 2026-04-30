#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout + 2d EMA50 trend filter + volume confirmation
# Donchian channels provide robust trend-following structure; 2d EMA50 filters for higher-timeframe trend alignment.
# Volume spike (2.0x 20-period average) confirms institutional participation.
# Uses 12h timeframe for entry timing, 1d/2d for signal direction. Discrete sizing 0.25 to minimize fee churn.
# Session filter (00-23 UTC) - no session filter for 12h as it captures major moves. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Donchian20_2dEMA50_VolumeConfirm_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period) using prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper/lower bands (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Load 2d data ONCE before loop for EMA50 trend filter
    df_2d = get_htf_data(prices, '2d')
    if len(df_2d) < 50:
        return np.zeros(n)
    
    # Calculate 2d EMA50
    close_2d = df_2d['close'].values
    ema_50_2d = pd.Series(close_2d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_2d_aligned = align_htf_to_ltf(prices, df_2d, ema_50_2d)
    
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
        curr_donchian_high = donchian_high_aligned[i]
        curr_donchian_low = donchian_low_aligned[i]
        curr_ema_50_2d = ema_50_2d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above Donchian upper band AND above 2d EMA50 (uptrend)
                if curr_close > curr_donchian_high and curr_close > curr_ema_50_2d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian lower band AND below 2d EMA50 (downtrend)
                elif curr_close < curr_donchian_low and curr_close < curr_ema_50_2d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below Donchian lower band or below 2d EMA50
            if curr_close < curr_donchian_low or curr_close < curr_ema_50_2d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian upper band or above 2d EMA50
            if curr_close > curr_donchian_high or curr_close > curr_ema_50_2d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals