#!/usr/bin/env python3
"""
Experiment #3827: 6h Donchian(20) breakout + 1d volume confirmation + 1w pivot direction
HYPOTHESIS: 6h Donchian breakouts capture medium-term swings. 1d volume (>1.5x average) confirms institutional participation. 1w pivot (R1/S1) provides directional bias: only long above weekly pivot, short below. Works in bull markets (breakouts above resistance with volume) and bear markets (breakdowns below support with volume). Discrete position sizing (0.25) minimizes fee drag. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3827_6h_donchian20_1d_vol_1w_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume MA ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume MA(20)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones(len(volume_1d))
    vol_ratio_1d[20:] = volume_1d[20:] / vol_ma_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === HTF: 1w data for pivot points (R1, S1, PP) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    
    # Align 1w pivot points to 6h timeframe (shifted by 1 for completed 1w bar)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(pp_1w_aligned[i]) or
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                # Calculate ATR(14) for trailing stop
                if i >= 14:
                    tr = np.maximum(high[i] - low[i], 
                                   np.maximum(np.abs(high[i] - close[i-1]), 
                                              np.abs(low[i] - close[i-1])))
                    atr_14 = np.mean([np.maximum(high[j] - low[j], 
                                                 np.maximum(np.abs(high[j] - close[j-1]), 
                                                            np.abs(low[j] - close[j-1]))) 
                                      for j in range(i-13, i+1)])
                    if price < highest_since_entry - 2.0 * atr_14:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if i >= 14:
                    tr = np.maximum(high[i] - low[i], 
                                   np.maximum(np.abs(high[i] - close[i-1]), 
                                              np.abs(low[i] - close[i-1])))
                    atr_14 = np.mean([np.maximum(high[j] - low[j], 
                                                 np.maximum(np.abs(high[j] - close[j-1]), 
                                                            np.abs(low[j] - close[j-1]))) 
                                      for j in range(i-13, i+1)])
                    if price > lowest_since_entry + 2.0 * atr_14:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) 
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND above weekly R1 (bullish breakout with volume confirmation)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                price > r1_1w_aligned[i]):     # Above weekly R1 (bullish bias)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below weekly S1 (bearish breakdown with volume confirmation)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < s1_1w_aligned[i]):     # Below weekly S1 (bearish bias)
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals