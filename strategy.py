#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v1
# Hypothesis: 4h strategy using Donchian(20) breakouts with volume confirmation (>1.5x 20-bar avg volume) and choppiness regime filter (CHOP(14) > 61.8 for ranging markets). Enters long when price breaks above Donchian upper channel with volume confirmation and chop > 61.8 (mean reversion setup). Enters short when price breaks below Donchian lower channel with volume confirmation and chop > 61.8. Uses discrete sizing (0.25) to limit fee churn. Target: 20-50 trades/year (80-200 total over 4 years). Choppiness filter ensures we only trade in ranging markets where mean reversion works, avoiding strong trends that cause false breakouts. Works in both bull and bear markets as ranging regimes persist across cycles.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) - measures ranging vs trending markets
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # First bar has no previous close
    atr1 = pd.Series(tr).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    atr_max = pd.Series(high - low).rolling(window=14, min_periods=14).max().values
    atr_min = pd.Series(high - low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (14 * np.log10(14))) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price touches Donchian lower channel (mean reversion target)
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches Donchian upper channel (mean reversion target)
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Donchian breakout with volume confirmation and chop filter
            bullish_breakout = (close[i] > donchian_high[i]) and volume_confirmed and chop_filter
            bearish_breakout = (close[i] < donchian_low[i]) and volume_confirmed and chop_filter
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals