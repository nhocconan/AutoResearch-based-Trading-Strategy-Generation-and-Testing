#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hVolumeSpike_RegimeFilter
Hypothesis: Camarilla R3/S3 breakout with 12h volume spike and choppiness regime filter.
Works in bull markets via trend-following breaks and in bear markets via volatility expansion mean reversion.
Volume spike ensures institutional participation, chop filter avoids ranging markets.
Target: 20-35 trades/year via tight confluence of three filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R3 and S3 levels (R3 = C + 1.1*(HL/2), S3 = C - 1.1*(HL/2))
    R3 = prev_close + 1.1 * prev_range * (1.0/2.0)
    S3 = prev_close - 1.1 * prev_range * (1.0/2.0)
    
    # Align 1d levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 12h volume spike: current volume > 2.0 * 20-period volume MA (using 12h data resampled conceptually)
    # Instead, we use volume ratio on 4h but require it to be above 12h average volume
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    vol_spike = volume > (2.0 * vol_ma_20_12h_aligned)
    
    # Choppiness regime filter: CHOP(14) < 61.8 = trending (we want to trade in trending markets)
    # Calculate True Range and ATR for chop calculation
    tr1 = np.maximum(high, np.roll(close, 1))
    tr1 = np.maximum(tr1, np.roll(low, 1))
    tr2 = np.abs(np.roll(close, 1) - low)
    tr3 = np.abs(np.roll(close, 1) - high)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(tr14) / (max_high14 - min_low14)) / log10(14)
    range_14 = max_high_14 - min_low_14
    chop = 100 * np.log10(sum_tr_14 / range_14) / np.log10(14)
    chop_filter = chop < 61.8  # Trending market regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (1) + 12h volume MA (20) + chop (14)
    start_idx = max(1, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals with all filters
            # Long breakout: price breaks above R3 with volume spike and trending regime
            long_breakout = (curr_close > R3_aligned[i]) and vol_spike[i] and chop_filter[i]
            # Short breakout: price breaks below S3 with volume spike and trending regime
            short_breakout = (curr_close < S3_aligned[i]) and vol_spike[i] and chop_filter[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below S3 (mean reversion) or regime changes to choppy
            if curr_close < S3_aligned[i] or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above R3 (mean reversion) or regime changes to choppy
            if curr_close > R3_aligned[i] or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hVolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0