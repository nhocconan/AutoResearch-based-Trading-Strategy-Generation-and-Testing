#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for signal direction (primary HTF)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Upper/lower Donchian bands
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_4h = align_htf_to_ltf(prices, df_4h, highest_20)
    donchian_lower_4h = align_htf_to_ltf(prices, df_4h, lowest_20)
    
    # === 4h Indicators: Volume Spike (20-bar avg) ===
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / (vol_ma_20 + 1e-10)
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # === 1d Indicators: Choppiness Index (14) for regime filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula
    chop_denominator = np.log10(atr_14 / (hh_14 - ll_14 + 1e-10)) * np.log10(14)
    chop = 100 * np.abs(chop_denominator) / np.log10(14)
    chop_1d = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]) or
            np.isnan(vol_ratio_4h_aligned[i]) or np.isnan(chop_1d[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (Chop > 61.8)
        if chop_1d[i] <= 61.8:
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above 4h Donchian upper with volume spike
        if (close[i] > donchian_upper_4h[i] and 
            vol_ratio_4h_aligned[i] > 1.5):
            signals[i] = 0.20
            
        # Short breakout: price breaks below 4h Donchian lower with volume spike
        elif (close[i] < donchian_lower_4h[i] and 
              vol_ratio_4h_aligned[i] > 1.5):
            signals[i] = -0.20
            
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4hDonchian_Volume_Chop_Session"
timeframe = "1h"
leverage = 1.0