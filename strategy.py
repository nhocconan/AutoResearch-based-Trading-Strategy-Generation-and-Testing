#!/usr/bin/env python3
"""
Experiment #2511: 6h Camarilla Pivot Fade/Breakout + Volume Confirmation
HYPOTHESIS: Camarilla pivot levels from 1d HTF provide institutional support/resistance zones.
Fade at R3/S3 (mean reversion in range) and breakout continuation at R4/S4 (trend acceleration).
Volume confirmation filters low-probability breakouts. Works in both bull/bear markets by
adapting to regime (range vs trend) via price action at pivot levels. Targets 50-150 trades
over 4 years with discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2511_6h_camarilla_pivot_v1"
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
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels
    r4_1d = close_1d + range_1d * 1.1 / 2
    r3_1d = close_1d + range_1d * 1.1 / 4
    s3_1d = close_1d - range_1d * 1.1 / 4
    s4_1d = close_1d - range_1d * 1.1 / 2
    
    # Align HTF levels to LTF (6h)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
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
    
    warmup = 50  # sufficient for HTF and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Trailing stop at 2*ATR using Donchian width proxy ---
        if in_position:
            # Calculate ATR proxy from recent 6h price action
            lookback = min(20, i)
            if lookback >= 2:
                recent_high = np.max(high[i-lookback:i+1])
                recent_low = np.min(low[i-lookback:i+1])
                atr_estimate = (recent_high - recent_low) * 0.15
            else:
                atr_estimate = 0.0
            
            if position_side > 0:  # Long
                # Exit if price drops 2*ATR below entry
                if price < entry_price - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches opposite Camarilla level (mean reversion)
                elif price >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit if price rises 2*ATR above entry
                if price > entry_price + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches opposite Camarilla level (mean reversion)
                elif price <= s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Fade at R3/S3: price rejects extreme levels (mean reversion in range)
            if abs(price - r3_aligned[i]) < (r4_aligned[i] - r3_aligned[i]) * 0.1:
                # Price near R3, expect reversal down
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            elif abs(price - s3_aligned[i]) < (s3_aligned[i] - s4_aligned[i]) * 0.1:
                # Price near S3, expect reversal up
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Breakout continuation at R4/S4: price breaks extreme levels with volume
            elif price > r4_aligned[i]:
                # Break above R4, expect continuation up
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif price < s4_aligned[i]:
                # Break below S4, expect continuation down
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals