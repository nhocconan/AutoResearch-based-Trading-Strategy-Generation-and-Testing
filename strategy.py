#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v2
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.8x 20-period average) and chop regime filter (CHOP(14) > 61.8 = ranging). 
# Long when price breaks above upper band in ranging market with volume spike. Short when breaks below lower band in ranging market with volume spike.
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 25-40 trades/year (~100-160 total over 4 years).
# Donchian provides structure, volume confirms conviction, chop filter avoids whipsaws in strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v2"
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
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) - ranging market filter
    # CHOP = 100 * log10(sum(ATR(1)) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First bar TR
    atr1 = pd.Series(tr).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr1 / (np.log10(14) * (max_high - min_low)))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        in_ranging_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price falls below lower Donchian band OR volume drops
            if close[i] < lowest_low[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above upper Donchian band OR volume drops
            if close[i] > highest_high[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and in_ranging_market:
                # Long entry: price breaks above upper Donchian band with volume spike
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below lower Donchian band with volume spike
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals