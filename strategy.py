#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_v1
Hypothesis: TRIX momentum + volume spike confirmation + choppiness regime filter on 4h. Uses TRIX(12) crossing zero for trend changes with volume > 2.5x average to confirm institutional participation. Chop(14) > 61.8 for ranging markets (mean reversion at extremes) and Chop < 38.2 for trending markets (breakout continuation). Designed for low trade frequency (target 20-50/year) with discrete sizing (0.25) to minimize fee drag. Works in bull markets via breakout continuation and in bear markets via mean reversion in ranges, adapting to regime automatically.
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
    
    # Calculate TRIX(12) - triple smoothed EMA rate of change
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.pct_change(periods=1))  # % change of triple smoothed EMA
    trix_values = trix.values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter: volume > 2.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    # Calculate Choppiness Index(14)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high - lowest_low
    # Avoid division by zero and log of zero
    chop_raw = np.where((range_14 > 0) & (atr_14 > 0), 
                        100 * np.log10(atr_14) / np.log10(range_14) / np.log10(14), 
                        50.0)  # neutral when undefined
    chop_values = chop_raw
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of TRIX warmup (~36), ATR(14), volume MA(20), chop(14)
    start_idx = max(36, 14, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_values[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(chop_values[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trix_val = trix_values[i]
        vol_spike = volume_spike[i]
        chop_val = chop_values[i]
        
        # Regime definitions
        is_ranging = chop_val > 61.8   # Chop > 61.8 = ranging/mean revert
        is_trending = chop_val < 38.2  # Chop < 38.2 = trending/breakout
        
        if position == 0:
            # Long conditions:
            # In trending market: TRIX crosses above zero with volume spike (breakout continuation)
            # In ranging market: TRIX crosses above zero from negative with volume spike (mean reversion long)
            trix_cross_up = (trix_values[i-1] <= 0) and (trix_val > 0)
            long_signal = trix_cross_up and vol_spike and (is_trending or is_ranging)
            
            # Short conditions:
            # In trending market: TRIX crosses below zero with volume spike (breakdown continuation)
            # In ranging market: TRIX crosses below zero from positive with volume spike (mean reversion short)
            trix_cross_down = (trix_values[i-1] >= 0) and (trix_val < 0)
            short_signal = trix_cross_down and vol_spike and (is_trending or is_ranging)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TRIX crosses below zero OR ATR stoploss hit
            trix_cross_down = (trix_values[i-1] >= 0) and (trix_val < 0)
            if trix_cross_down or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR ATR stoploss hit
            trix_cross_up = (trix_values[i-1] <= 0) and (trix_val > 0)
            if trix_cross_up or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0