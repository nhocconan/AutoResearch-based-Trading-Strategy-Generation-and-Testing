#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian breakout + volume confirmation
# Donchian(20) on 1w provides clear weekly trend direction and structure
# Long when daily price breaks above 1w Donchian upper with volume confirmation
# Short when daily price breaks below 1w Donchian lower with volume confirmation
# Uses discrete position sizing 0.25 to target ~15-25 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows trends, volume filter avoids false breakouts

name = "1d_1w_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_upper_20 = rolling_max(high_1w, 20)
    donchian_lower_20 = rolling_min(low_1w, 20)
    
    # Calculate 1w average volume (20-period)
    vol_s_1w = pd.Series(volume_1w)
    avg_vol_1w = vol_s_1w.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_20)
    avg_vol_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_vol_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(avg_vol_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume (20-period)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
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
            # Breakout strategy: enter on Donchian breakout with volume confirmation
            if close[i] > donchian_upper_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_lower_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals