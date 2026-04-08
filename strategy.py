#!/usr/bin/env python3
# 12h_1d_adx_volume_v1
# Hypothesis: 12h price breaking above/below daily Donchian(20) channels with ADX>25
# and volume > 1.5x average creates high-probability trend continuation trades.
# Uses 1d timeframe for Donchian channels and ADX calculation (proper trend strength)
# and 12h for entry timing. Works in bull/bear markets by trading breakouts in
# direction of prevailing trend (ADX>25 indicates strong trend). Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Get 1d data for Donchian channels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper channel: highest high of last 20 days
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 days
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-period) on 1d data
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, ignore_na=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, ignore_na=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, ignore_na=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, ignore_na=False).mean().values
    
    # Align 1d indicators to 12h timeframe
    upper_channel_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x average of last 24 periods (12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(upper_channel_1d_aligned[i]) or np.isnan(lower_channel_1d_aligned[i]) or \
           np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below lower channel or trend weakens (ADX < 20)
            if close[i] < lower_channel_1d_aligned[i] or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above upper channel or trend weakens (ADX < 20)
            if close[i] > upper_channel_1d_aligned[i] or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above upper channel with ADX>25 and volume
            if (close[i] > upper_channel_1d_aligned[i] and 
                open_prices[i] <= upper_channel_1d_aligned[i] and  # Ensure breakout happened this bar
                adx_1d_aligned[i] > 25 and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower channel with ADX>25 and volume
            elif (close[i] < lower_channel_1d_aligned[i] and 
                  open_prices[i] >= lower_channel_1d_aligned[i] and  # Ensure breakdown happened this bar
                  adx_1d_aligned[i] > 25 and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals