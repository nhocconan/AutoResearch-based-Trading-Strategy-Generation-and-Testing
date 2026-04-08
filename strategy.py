#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_spike_v1
# Hypothesis: Camarilla pivot levels from daily timeframe act as intraday support/resistance on 4h chart.
# Long when price breaks above H3 with volume spike; short when breaks below L3 with volume spike.
# Use choppiness regime filter to avoid false breakouts in ranging markets.
# Designed to work in both bull (breakouts) and bear (mean reversion at extremes) markets.
# Volume confirmation ensures institutional participation; chop filter avoids whipsaws.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_spike_v1"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align to 4h timeframe (completed daily bars only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume spike detection (20-period volume average)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count >= 20:
            vol_ma[i] = vol_sum / vol_count
            vol_sum -= volume[i - 19]
            vol_count -= 1
    
    volume_spike = np.zeros(n, dtype=bool)
    for i in range(20, n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            volume_spike[i] = volume[i] > 2.0 * vol_ma[i]
    
    # Choppiness regime filter (14-period)
    chop = np.full(n, np.nan)
    for i in range(14, n):
        highest_high = np.max(high[i-14:i+1])
        lowest_low = np.min(low[i-14:i+1])
        atr_sum = 0.0
        for j in range(i-14, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        atr = atr_sum / 15
        if atr > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
        else:
            chop[i] = 50.0
    
    # Choppiness regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    # We use CHOP < 50 to avoid strong trends where breakouts fail
    chop_regime = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(chop[i]):
            chop_regime[i] = chop[i] < 50.0  # Prefer trending to ranging for breakouts
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price drops below H3 or momentum fails
            if close[i] < h3_aligned[i] or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above L3 or momentum fails
            if close[i] > l3_aligned[i] or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 with volume spike in trending regime
            if close[i] > h3_aligned[i] and volume_spike[i] and chop_regime[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 with volume spike in trending regime
            elif close[i] < l3_aligned[i] and volume_spike[i] and chop_regime[i]:
                position = -1
                signals[i] = -0.25
    
    return signals