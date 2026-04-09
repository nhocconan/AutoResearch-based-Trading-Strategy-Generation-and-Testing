#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v2
# Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation (>1.5x 20-period average volume) and chop regime filter (CHOP(14) > 61.8 = ranging market for mean reversion). Enters long when price breaks above Donchian upper band with volume confirmation in chop regime; short when price breaks below Donchian lower band with volume confirmation in chop regime. Exits when price returns to Donchian midpoint. Designed for low turnover (target: 20-50 trades/year) to capture mean reversion in ranging markets while avoiding trending markets where breakouts fail. Works in both bull and bear markets by focusing on ranging regimes where price respects channels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma1 = pd.Series(series).ewm(span=half_period, adjust=False, min_periods=half_period).mean()
    wma2 = pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean()
    return hma.values

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    atr = pd.Series(np.maximum(np.abs(high - low), np.maximum(np.abs(high - close.shift()), np.abs(low - close.shift())))).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

name = "4h_donchian_breakout_volume_chop_regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Chop regime filter (14-period)
    chop = calculate_chop(high, low, close, 14)
    chop_regime = chop > 61.8  # Chop > 61.8 = ranging market (mean reversion favorable)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and chop regime (ranging market)
            if volume_confirmed and chop_regime[i]:
                # Long: price breaks above Donchian upper band
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower band
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals