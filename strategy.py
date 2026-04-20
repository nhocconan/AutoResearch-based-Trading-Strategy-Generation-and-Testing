#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime + Williams %R mean reversion
# Choppiness Index (14) > 61.8 identifies ranging markets (mean revert)
# Williams %R(14) < -80 = oversold, > -20 = overbought
# Entry: Long when Williams %R crosses above -80 in ranging market
# Short when Williams %R crosses below -20 in ranging market
# Exit: Williams %R crosses above -50 (long) or below -50 (short)
# Designed for 4h timeframe with regime filter to reduce false signals
# Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14)
    atr_list = []
    for i in range(n):
        if i < 14:
            atr_list.append(np.nan)
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr_list.append(tr)
    atr = np.array(atr_list)
    
    # True Range for CHOP calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of ATR over 14 periods
    atr_sum = np.zeros(n)
    for i in range(n):
        if i < 14:
            atr_sum[i] = np.nan
        else:
            atr_sum[i] = np.nansum(atr[i-13:i+1])
    
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.zeros(n)
    for i in range(n):
        if i < 13 or np.isnan(atr_sum[i]) or np.isnan(max_high[i]) or np.isnan(min_low[i]) or max_high[i] == min_low[i]:
            chop[i] = np.nan
        else:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
    
    # Williams %R (14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    wilr = -100 * (highest_high - close) / (highest_high - lowest_low)
    wilr[highest_high == lowest_low] = np.nan
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if NaN in indicators
        if np.isnan(chop[i]) or np.isnan(wilr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop > 61.8 = ranging market (mean revert)
        is_ranging = chop[i] > 61.8
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        if position == 0:
            # Long entry: Williams %R crosses above -80 in ranging market
            long_signal = (wilr[i] > -80) and (wilr[i-1] <= -80) and is_ranging and has_volume
            
            # Short entry: Williams %R crosses below -20 in ranging market
            short_signal = (wilr[i] < -20) and (wilr[i-1] >= -20) and is_ranging and has_volume
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50
            if wilr[i] > -50 and wilr[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50
            if wilr[i] < -50 and wilr[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ChopRegime_WilliamsR_MeanRev"
timeframe = "4h"
leverage = 1.0