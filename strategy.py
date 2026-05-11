#!/usr/bin/env python3
name = "4h_Donchian20_Upper_Lower_Breakout_Trend_Filter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily trend filter: EMA34 on daily close (must be in uptrend for long, downtrend for short)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Donchian channel (20-period) on 4h
    # Use pandas rolling with min_periods to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.3 * 20-period SMA of volume (on 4h)
    vol_series = pd.Series(volume)
    vol_sma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.3 * vol_sma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure indicators are warmed up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_sma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band + uptrend on 1d and 1w + volume confirmation
            if (close[i] > donchian_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band + downtrend on 1d and 1w + volume confirmation
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit if price breaks below Donchian lower band or trend turns down
            if close[i] < donchian_low[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above Donchian upper band or trend turns up
            if close[i] > donchian_high[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals