#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period avg + CHOP > 61.8 (range)
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period avg + CHOP > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# CHOP filter ensures we only trade in ranging markets where mean reversion works well.
# Volume confirmation ensures breakouts have conviction.
# Designed to work in both bull and bear markets by focusing on range-bound periods.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Donchian Channels (20-period) ===
    # Upper band: highest high over last 20 periods
    # Lower band: lowest low over last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Choppiness Index (CHOP) - 14 period ===
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    # Where ATR = True Range, n = period
    # High CHOP (>61.8) indicates ranging market
    # Low CHOP (<38.2) indicates trending market
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Max high and min low over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.full(n, np.nan)
    for i in range(n):
        if not (np.isnan(sum_atr[i]) or np.isnan(max_high[i]) or np.isnan(min_low[i]) or max_high[i] == min_low[i]):
            chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(14)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14) + 5  # Donchian(20) + CHOP(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(chop[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Choppiness filter: CHOP > 61.8 (ranging market)
        chop_filter = chop[i] > 61.8
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian high (close > upper band)
        # 2. Volume confirmation
        # 3. Chop filter (ranging market)
        if (close[i] > donchian_high[i]) and vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian low (close < lower band)
        # 2. Volume confirmation
        # 3. Chop filter (ranging market)
        elif (close[i] < donchian_low[i]) and vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0