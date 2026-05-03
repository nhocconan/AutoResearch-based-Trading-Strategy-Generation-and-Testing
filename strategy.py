#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and choppiness regime filter
# Long when price breaks above Donchian(20) high with volume > 1.5x 24-bar average and CHOP(14) > 61.8 (range regime)
# Short when price breaks below Donchian(20) low with volume > 1.5x 24-bar average and CHOP(14) > 61.8 (range regime)
# Exit when price crosses Donchian(20) midpoint or volume drops below average
# Donchian provides clear breakout levels, volume confirms institutional interest, chop filter avoids whipsaws in trends
# Target: 75-200 total trades over 4 years = 19-50/year. Uses discrete sizing (0.30) to balance return and fees.

name = "4h_Donchian_Breakout_Volume_Chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    midpoint = (highest_high + lowest_low) / 2
    
    # Volume confirmation (1.5x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Choppiness Index (14) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (log10(highest_high - lowest_low) * lookback)) / log10(lookback)
    tr1 = pd.Series(high).rolling(2).apply(lambda x: x[1] - x[0], raw=True).abs()
    tr2 = pd.Series(low).rolling(2).apply(lambda x: x[1] - x[0], raw=True).abs()
    tr3 = abs(pd.Series(high).shift(1) - pd.Series(low))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(highest_high_14 - lowest_low_14) * 14
    chop = np.where(
        (chop_denom > 0) & (atr_sum > 0),
        100 * np.log10(atr_sum) / chop_denom,
        50.0  # neutral when undefined
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(lookback, 24, 14) + 1
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: break above Donchian high with volume spike and chop > 61.8 (range regime)
            if (close[i] > highest_high[i] and 
                volume_spike[i] and chop[i] > 61.8):
                signals[i] = 0.30
                position = 1
            # Short entry: break below Donchian low with volume spike and chop > 61.8 (range regime)
            elif (close[i] < lowest_low[i] and 
                  volume_spike[i] and chop[i] > 61.8):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price below midpoint or volume drops below average
            if (close[i] < midpoint[i] or 
                volume[i] <= vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price above midpoint or volume drops below average
            if (close[i] > midpoint[i] or 
                volume[i] <= vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals