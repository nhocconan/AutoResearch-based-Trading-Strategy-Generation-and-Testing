#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and Choppiness index regime filter
# Uses 12h price breaking above/below 20-period Donchian channel for entry, confirmed by 1d volume spike
# and filtered by 1d Choppiness index to avoid ranging markets. Designed to capture strong trends
# while avoiding false breakouts in chop. Target: 15-25 trades/year.
name = "12h_Donchian20_1dVolume_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and Choppiness filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume SMA(20) for spike detection
    vol_1d = df_1d['volume'].values
    vol_sma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    # Calculate 1d Choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of absolute price changes over 14 periods
    abs_diff = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    sum_abs_diff = pd.Series(abs_diff).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness index = 100 * log10(sum(abs_diff)/atr*sqrt(14)) / log10(sqrt(14))
    chop = 100 * np.log10(sum_abs_diff / (atr * np.sqrt(14))) / np.log10(np.sqrt(14))
    chop = np.where((atr > 0) & (sum_abs_diff > 0), chop, 50)  # default to 50 if invalid
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high over 20 periods
    upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_sma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_band = upper_aligned[i]
        lower_band = lower_aligned[i]
        vol_sma = vol_sma_20_aligned[i]
        chop_val = chop_aligned[i]
        
        # Volume spike condition: current volume > 1.5 * 20-period average
        vol_spike = volume[i] > (1.5 * vol_sma)
        # Chop filter: only trade when market is trending (CHOP < 38.2) or in extreme chop (CHOP > 61.8 for mean reversion)
        # For breakout strategy, we prefer trending markets: CHOP < 38.2
        trending_market = chop_val < 38.2
        
        if position == 0:
            # Enter long: price breaks above upper Donchian band + volume spike + trending market
            if price > upper_band and vol_spike and trending_market:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian band + volume spike + trending market
            elif price < lower_band and vol_spike and trending_market:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below lower Donchian band OR chop becomes too high (ranging market)
            if price < lower_band or chop_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper Donchian band OR chop becomes too high (ranging market)
            if price > upper_band or chop_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals