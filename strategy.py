#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Breakout_With_Volume_and_Chop_Regime_v1
Hypothesis: On 4h timeframe, buy when price breaks above Camarilla H3 level with volume spike in trending regime,
sell when price breaks below L3 level with volume spike in trending regime. Uses 12h Camarilla pivot levels
for structure and 12h Choppiness Index to filter ranging markets. Designed for 20-40 trades/year by requiring
trending regime (CHOP < 38.2) and volume confirmation, avoiding false breakouts in sideways markets.
Works in bull markets via long breakouts and in bear markets via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Pivot_Breakout_With_Volume_and_Chop_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Camarilla pivot levels and Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (using previous 12h bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: based on previous period's range
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), H2 = close + 0.5*(high-low)
    # L2 = close - 0.5*(high-low), L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    range_12h = high_12h - low_12h
    camarilla_h3 = close_12h + 1.0 * range_12h  # H3 level
    camarilla_l3 = close_12h - 1.0 * range_12h  # L3 level
    
    # Align Camarilla levels to 4h timeframe (wait for 12h bar to close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # Calculate 12h Choppiness Index for regime filtering
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    period = 14
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = 0  # first value has no previous close
    tr3[0] = 0
    atr_12h = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Sum of ATR over period
    atr_sum = np.zeros_like(atr_12h)
    for i in range(len(atr_12h)):
        if i < period:
            atr_sum[i] = np.nan
        else:
            atr_sum[i] = np.sum(atr_12h[i-period+1:i+1])
    
    # Rolling max/high and min/low over period
    max_high = np.zeros_like(high_12h)
    min_low = np.zeros_like(low_12h)
    for i in range(len(high_12h)):
        if i < period:
            max_high[i] = np.nan
            min_low[i] = np.nan
        else:
            max_high[i] = np.max(high_12h[i-period+1:i+1])
            min_low[i] = np.min(low_12h[i-period+1:i+1])
    
    # Choppiness Index
    chop_12h = np.zeros_like(close_12h)
    for i in range(len(chop_12h)):
        if np.isnan(atr_sum[i]) or np.isnan(max_high[i]) or np.isnan(min_low[i]) or max_high[i] == min_low[i]:
            chop_12h[i] = np.nan
        else:
            chop_12h[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
    
    # Align Choppiness Index to 4h timeframe
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Volume average (20 period) for spike detection on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: trending market (CHOP < 38.2)
        trending_regime = chop_12h_aligned[i] < 38.2
        
        # Volume spike: current volume > 2.0x average
        volume_spike = volume[i] > vol_ma[i] * 2.0
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # Entry conditions: breakout + volume spike + trending regime
        long_entry = long_breakout and volume_spike and trending_regime
        short_entry = short_breakout and volume_spike and trending_regime
        
        # Exit conditions: price returns to pivot zone (between H3 and L3) or opposite breakout
        pivot_zone_entry = camarilla_l3_aligned[i] <= close[i] <= camarilla_h3_aligned[i]
        long_exit = pivot_zone_entry or short_breakout
        short_exit = pivot_zone_entry or long_breakout
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals