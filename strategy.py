#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and chop regime filter
# Long when price breaks above Donchian(20) high + volume > 1.8x 20-period volume SMA + chop < 61.8 (trending)
# Short when price breaks below Donchian(20) low + volume > 1.8x 20-period volume SMA + chop < 61.8 (trending)
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
# Donchian breakouts capture strong trends; volume confirmation reduces false signals; chop filter ensures trending markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop for chop regime calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Chopiness Index (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.maximum(high_1d - low_1d,
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # first period
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(tr) / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(14)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Indicators: Donchian Channels (20-period) ===
    # Donchian upper = max(high, 20)
    # Donchian lower = min(low, 20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need chop(14) + donchian(20) + volume(20) + buffer
    warmup = 40
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # Chop filter: only trade when trending (chop < 61.8)
        trending = chop_aligned[i] < 61.8
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian(20) high
        # 2. Volume confirmation
        # 3. Trending market (chop < 61.8)
        if (close[i] > donchian_high[i]) and \
           vol_confirm and trending:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian(20) low
        # 2. Volume confirmation
        # 3. Trending market (chop < 61.8)
        elif (close[i] < donchian_low[i]) and \
             vol_confirm and trending:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0