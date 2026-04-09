#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout + volume confirmation + chop regime filter
# Weekly Donchian(20) provides major trend structure and breakout signals
# Long when price breaks above weekly Donchian upper with volume confirmation and chop < 61.8 (trending regime)
# Short when price breaks below weekly Donchian lower with volume confirmation and chop < 61.8
# Uses discrete position sizing 0.25 to target ~15-25 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows major trends, chop filter avoids whipsaws in ranging markets

name = "1d_1w_donchian_breakout_v2"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_upper_20 = rolling_max(high_1w, 20)
    donchian_lower_20 = rolling_min(low_1w, 20)
    
    # Calculate weekly average volume (20-period)
    vol_s_1w = pd.Series(volume_1w)
    avg_vol_1w = vol_s_1w.rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_20)
    avg_vol_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_vol_1w)
    
    # Calculate daily chopiness index (14-period) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high, low, prev_close)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chopiness index: 100 * log10(sum(TR14)/(log10(14)*(HH14-LL14)))
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = np.log10(14) * (hh_14 - ll_14)
    chop = np.where(denominator != 0, 100 * (np.log10(sum_tr_14) / denominator), 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(avg_vol_1w_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume (20-period)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Chop regime filter: trending market (chop < 61.8)
        trending_regime = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit long if price falls below weekly Donchian lower
            if close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above weekly Donchian upper
            if close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on weekly Donchian breakout with volume confirmation and trending regime
            if close[i] > donchian_upper_aligned[i] and volume_confirmed and trending_regime:
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_lower_aligned[i] and volume_confirmed and trending_regime:
                position = -1
                signals[i] = -0.25
    
    return signals