#!/usr/bin/env python3
"""
Experiment #3831: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture medium-term swings with 1d volume (>1.8x) confirming institutional participation. 
Camarilla pivot levels from 1d provide precise entry/exit zones: fade at R3/S3, breakout continuation at R4/S4. 
Works in bull markets (breakouts above R4) and bear markets (breakdowns below S4). Discrete position sizing (0.25) minimizes fee drag. 
Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3831_6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    camarilla_h4 = np.full(len(close_1d), np.nan)  # R4
    camarilla_l4 = np.full(len(close_1d), np.nan)  # S4
    camarilla_h3 = np.full(len(close_1d), np.nan)  # R3
    camarilla_l3 = np.full(len(close_1d), np.nan)  # S3
    
    for i in range(len(close_1d)):
        if i < 1:
            continue
        # Camarilla pivot calculation
        pivot = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3
        range_ = high_1d[i-1] - low_1d[i-1]
        camarilla_h4[i] = close_1d[i-1] + range_ * 1.1 / 2
        camarilla_l4[i] = close_1d[i-1] - range_ * 1.1 / 2
        camarilla_h3[i] = close_1d[i-1] + range_ * 1.1 / 4
        camarilla_l3[i] = close_1d[i-1] - range_ * 1.1 / 4
    
    # Align 1d Camarilla levels to 6h timeframe (shifted by 1 for completed 1d bar)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
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
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                # Calculate ATR manually for exit condition
                if i > 0:
                    atr_temp = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                    if price < highest_since_entry - 2.0 * atr_temp:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if price breaks below Donchian lower band (trend reversal)
                    elif price < lowest_low[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if price reaches Camarilla L3 (take profit for long)
                    elif price <= l3_1d_aligned[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if i > 0:
                    atr_temp = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                    if price > lowest_since_entry + 2.0 * atr_temp:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if price breaks above Donchian upper band (trend reversal)
                    elif price > highest_high[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if price reaches Camarilla H3 (take profit for short)
                    elif price >= h3_1d_aligned[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND above Camarilla H4 (bullish breakout with volume confirmation)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                price > h4_1d_aligned[i]):     # Above Camarilla H4 (strong bullish breakout)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below Camarilla L4 (bearish breakdown with volume confirmation)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < l4_1d_aligned[i]):     # Below Camarilla L4 (strong bearish breakdown)
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