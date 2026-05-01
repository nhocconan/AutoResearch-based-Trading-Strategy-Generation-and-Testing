#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses price channel breakouts for trend continuation, filtered by daily EMA trend and volume.
# Works in bull markets (buy breakouts with uptrend) and bear markets (sell breakdowns with downtrend).
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
# Discrete position sizing (0.25) to minimize fee churn.

name = "12h_Donchian20_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Donchian channels from previous day OHLC (20-day high/low)
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 20:
        return np.zeros(n)
    
    # 20-day high and low for Donchian channels
    high_20 = pd.Series(df_1d_ohlc['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d_ohlc['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d_ohlc, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d_ohlc, low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA34, volume median, and Donchian channels
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(high_20_aligned[i]) or
            np.isnan(low_20_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Donchian breakout conditions (20-day channel breakout)
        breakout_up = curr_close > high_20_aligned[i]   # break above 20-day high
        breakout_down = curr_close < low_20_aligned[i]  # break below 20-day low
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend AND volume confirmation
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakout down AND downtrend AND volume confirmation
            elif breakout_down and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakout down (reversal signal)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout up (reversal signal)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals