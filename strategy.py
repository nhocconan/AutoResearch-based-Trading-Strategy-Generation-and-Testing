#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume confirmation + choppiness regime filter on 1h timeframe.
Long when price breaks above 4h Donchian upper band AND volume > 1.5x 4h average AND 1d chop < 61.8 (trending).
Short when price breaks below 4h Donchian lower band AND volume > 1.5x 4h average AND 1d chop < 61.8.
Exit when price reverts to 4h Donchian middle (20-period mean) OR 1d chop > 61.8 (choppy market).
Uses 4h for Donchian calculation and 1d for choppiness filter to reduce whipsaw and avoid overtrading.
Target: 60-150 total trades over 4 years (15-37/year). Discrete position sizing (0.0, ±0.20) to minimize fee churn.
Session filter (08-20 UTC) applied to reduce noise trades outside active market hours.
Works in bull markets (captures uptrends) and bear markets (captures downtrends) by following 4h structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (UTC 08-20)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian calculation (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels on 4h timeframe (20-period)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_4h_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = ((donchian_upper + donchian_lower) / 2).values
    
    # Get 1d data for choppiness filter (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate choppiness index on 1d timeframe (14-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = high_1d_series.rolling(window=14, min_periods=14).max().values
    ll = low_1d_series.rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr)/log(hh/ll)) / log10(14)
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    ratio = hh / ll
    ratio = np.where(ratio <= 1, 1.001, ratio)  # avoid division by zero or log<=0
    chop = 100 * (np.log10(sum_atr) - np.log10(ratio)) / np.log10(14)
    
    # Align 4h Donchian to 1h timeframe (waits for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    # Align 1d chop to 1h timeframe (waits for completed 1d bar)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average (20-period) on 4h, aligned to 1h
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if not in trading session or any required data is not available
        if not in_session[i] or \
           (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        du = donchian_upper_aligned[i]
        dl = donchian_lower_aligned[i]
        dm = donchian_middle_aligned[i]
        chop_val = chop_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Donchian upper AND volume > 1.5x avg AND chop < 61.8 (trending)
            if price > du and vol > 1.5 * vol_ma and chop_val < 61.8:
                signals[i] = 0.20
                position = 1
            # Short: price < Donchian lower AND volume > 1.5x avg AND chop < 61.8 (trending)
            elif price < dl and vol > 1.5 * vol_ma and chop_val < 61.8:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian middle OR chop > 61.8 (choppy market)
            if price < dm or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price > Donchian middle OR chop > 61.8 (choppy market)
            if price > dm or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_Volume_Chop_Session_Filter"
timeframe = "1h"
leverage = 1.0