#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and choppiness regime filter
# Donchian breakout provides clear trend structure. Volume confirmation reduces false breakouts.
# Choppiness filter (CHOP > 61.8) avoids trending markets where mean reversion works better.
# Target: 12-37 trades/year on BTC/ETH/SOL with discrete position sizing 0.25 to minimize fee drag.
# Works in bull/bear markets: breakout follows trends, chop filter avoids choppy false signals.

name = "12h_1d_donchian_breakout_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_upper_20 = rolling_max(high_1d, 20)
    donchian_lower_20 = rolling_min(low_1d, 20)
    
    # Calculate 1d average volume (20-period)
    vol_s_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (14-period)
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    close_prev_1d = np.roll(close_1d, 1)
    close_prev_1d[0] = close_1d[0]
    tr_1d = true_range(high_1d, low_1d, close_prev_1d)
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    max_hh_1d = rolling_max(high_1d, 14)
    min_ll_1d = rolling_min(low_1d, 14)
    chop_denom = np.where(max_hh_1d - min_ll_1d == 0, 1e-10, max_hh_1d - min_ll_1d)
    chop_1d = 100 * np.log10(atr_1d * np.sqrt(14) / chop_denom) / np.log10(14)
    
    # Align 1d indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_20)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume (20-period)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Choppiness regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit long if price falls below Donchian lower
            if close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above Donchian upper
            if close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on Donchian breakout with volume confirmation and chop filter
            if close[i] > donchian_upper_aligned[i] and volume_confirmed and chop_filter:
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_lower_aligned[i] and volume_confirmed and chop_filter:
                position = -1
                signals[i] = -0.25
    
    return signals