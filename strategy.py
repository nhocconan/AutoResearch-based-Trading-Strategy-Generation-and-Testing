#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian breakout + volume confirmation + chop regime filter
# Donchian(20) on 1d provides clear trend direction and structure
# Long when price breaks above 1d Donchian upper with volume confirmation and chop < 61.8 (trending regime)
# Short when price breaks below 1d Donchian lower with volume confirmation and chop < 61.8
# Uses discrete position sizing 0.25 to target ~12-30 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows trends, chop filter avoids whipsaws in ranging markets

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
    
    # Calculate 1d Chopiness Index (14-period) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]
    tr_1d = true_range(high_1d, low_1d, prev_close_1d)
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate max/min high/low over 14 periods for Chop
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (max_high_14 - min_low_14)) / log10(14)
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    denominator = max_high_14 - min_low_14
    chop_1d = np.where(
        denominator > 0,
        100 * np.log10(sum_tr_14 / denominator) / np.log10(14),
        50  # neutral value when denominator is 0
    )
    
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
        
        # Regime filter: only trade when chop < 61.8 (trending market)
        trending_regime = chop_1d_aligned[i] < 61.8
        
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
            # Breakout strategy: enter on Donchian breakout with volume confirmation and trending regime
            if close[i] > donchian_upper_aligned[i] and volume_confirmed and trending_regime:
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_lower_aligned[i] and volume_confirmed and trending_regime:
                position = -1
                signals[i] = -0.25
    
    return signals