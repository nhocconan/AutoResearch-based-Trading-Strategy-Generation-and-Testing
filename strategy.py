#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout + volume confirmation + choppiness regime filter
# Long when price breaks above 12h Donchian upper (20) with volume confirmation and chop < 61.8 (trending)
# Short when price breaks below 12h Donchian lower (20) with volume confirmation and chop < 61.8 (trending)
# Exit on opposite Donchian breakout. Uses discrete size 0.25 to target ~25-40 trades/year.
# Works in bull/bear: breakout follows trends, chop filter avoids ranging markets, volume confirms strength.

name = "4h_12h_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_upper_20 = rolling_max(high_12h, 20)
    donchian_lower_20 = rolling_min(low_12h, 20)
    
    # Calculate 12h average volume (20-period)
    vol_s_12h = pd.Series(volume_12h)
    avg_vol_12h = vol_s_12h.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h ATR (14-period) for choppiness index
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h choppiness index (14-period)
    sum_tr_14 = pd.Series(atr_12h * 14).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(sum_tr_14 / (max_hh - min_ll)) / np.log10(14)
    chop_12h = 100 * chop_denom
    
    # Align 12h indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_20)
    avg_vol_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_vol_12h)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(avg_vol_12h_aligned[i]) or np.isnan(chop_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume (20-period)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Choppiness regime: trending when chop < 61.8
        chop_filter = chop_12h_aligned[i] < 61.8
        
        if position == 1:  # Long position
            # Exit long if price falls below 12h Donchian lower
            if close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above 12h Donchian upper
            if close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on 12h Donchian breakout with volume confirmation and trending regime
            if close[i] > donchian_upper_aligned[i] and volume_confirmed and chop_filter:
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_lower_aligned[i] and volume_confirmed and chop_filter:
                position = -1
                signals[i] = -0.25
    
    return signals