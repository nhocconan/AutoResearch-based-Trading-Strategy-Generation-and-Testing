#!/usr/bin/env python3
# 12h_adaptive_breakout_1d_trend_volume_v2
# Hypothesis: Adaptive Donchian breakout on 12h with 1d EMA trend filter and volume confirmation.
# Uses dynamic lookback period (15-25) based on volatility regime to reduce whipsaws.
# Long when price breaks above adaptive Donchian upper band with uptrend (price > 1d EMA50) and volume > 1.3x average.
# Short when price breaks below adaptive Donchian lower band with downtrend (price < 1d EMA50) and volume > 1.3x average.
# Exit when price crosses back to adaptive Donchian mid band.
# Designed to capture strong breakouts with trend alignment while minimizing false signals in choppy markets.
# Target: 60-120 total trades over 4 years (~15-30/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_adaptive_breakout_1d_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter (shorter for better responsiveness)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volatility-adjusted Donchian channel period
    # Use ATR-based volatility to adjust lookback: higher vol = shorter lookback, lower vol = longer lookback
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Normalize ATR to get volatility regime (0 to 1 range)
    atr_max = pd.Series(atr).rolling(window=100, min_periods=100).max().values
    atr_min = pd.Series(atr).rolling(window=100, min_periods=100).min().values
    # Avoid division by zero
    atr_range = atr_max - atr_min
    vol_regime = np.where(atr_range > 0, (atr - atr_min) / atr_range, 0.5)
    
    # Adaptive lookback: 15 in high volatility, 25 in low volatility
    lookback = np.round(15 + vol_regime * 10).astype(int)
    
    # Calculate adaptive Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        lb = max(1, lookback[i]) if not np.isnan(lookback[i]) else 20
        start_idx = max(0, i - lb + 1)
        if start_idx <= i:
            donchian_high[i] = np.max(high[start_idx:i+1])
            donchian_low[i] = np.min(low[start_idx:i+1])
    
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian middle band
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian middle band
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.3x average volume (reduced from 1.5 to increase signals slightly)
            volume_ok = volume[i] > 1.3 * avg_volume[i]
            
            # Breakout entries: Donchian upper breakout (long) and lower breakdown (short)
            if (close[i] > donchian_high[i]) and (close[i] > ema_50_1d_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (close[i] < donchian_low[i]) and (close[i] < ema_50_1d_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals