#!/usr/bin/env python3
"""
4h Donchian Breakout with 12h Trend Filter and Volume Confirmation
Long when price breaks above 4h Donchian upper channel (20) and 12h EMA21 > EMA50 (uptrend)
Short when price breaks below 4h Donchian lower channel (20) and 12h EMA21 < EMA50 (downtrend)
Exit when price crosses the midline (mean of upper/lower channel)
Volume filter: require current volume > 1.5x 20-period average volume
Designed to capture strong trends with filtered breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # === 4h Donchian Channel (20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === 12h EMA Trend Filter (21, 50) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Volume Filter (20-period average) ===
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_21_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_ok = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midline
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midline
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: EMA21 > EMA50 for long, EMA21 < EMA50 for short
            if ema_21_12h_aligned[i] > ema_50_12h_aligned[i]:
                # Uptrend - look for long breakout
                if close[i] > donchian_high[i] and volume_ok:
                    position = 1
                    signals[i] = 0.25
            elif ema_21_12h_aligned[i] < ema_50_12h_aligned[i]:
                # Downtrend - look for short breakdown
                if close[i] < donchian_low[i] and volume_ok:
                    position = -1
                    signals[i] = -0.25
    
    return signals