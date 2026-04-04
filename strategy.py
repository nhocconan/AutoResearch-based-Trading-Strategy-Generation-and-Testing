#!/usr/bin/env python3
"""
Experiment #3019: 6h Camarilla Pivot Fade/Breakout with Volume Confirmation
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion fade, R4/S4 for breakout) derived from 1d candles provide institutional support/resistance. On 6h timeframe: fade at R3/S3 when price reverts to mean (PPT), breakout continuation when price closes beyond R4/S4 with volume confirmation (>1.5x 20-period average). This strategy works in both bull/bear markets by adapting to price action at key levels. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3019_6h_camarilla_pivot_fade_breakout_vol_v1"
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
    
    # Calculate Camarilla pivot levels for each 1d bar
    # R4 = close + 1.1*(high - low)
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    # S4 = close - 1.1*(high - low)
    # PPT = (high + low + close)/3
    rng = high_1d - low_1d
    r4 = close_1d + 1.1 * rng
    r3 = close_1d + 1.1 * rng / 2
    s3 = close_1d - 1.1 * rng / 2
    s4 = close_1d - 1.1 * rng
    ppt = (high_1d + low_1d + close_1d) / 3
    
    # Align HTF levels to LTF (6h) with shift(1) for completed bars only
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    ppt_6h = align_htf_to_ltf(prices, df_1d, ppt)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
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
    
    warmup = 50  # sufficient for volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(s4_6h[i]) or np.isnan(ppt_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                atr_estimate = (high[i] - low[i]) * 0.5
                if price < highest_since_entry - 2.5 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches PPT (mean reversion target) or breaks S4 (stop)
                elif price >= ppt_6h[i] or price <= s4_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                atr_estimate = (high[i] - low[i]) * 0.5
                if price > lowest_since_entry + 2.5 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches PPT (mean reversion target) or breaks R4 (stop)
                elif price <= ppt_6h[i] or price >= r4_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average)
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Fade at R3/S3: price reverts to mean (PPT)
            # Short fade at R3: price rejects R3 and moves down toward PPT
            if price < r3_6h[i] and price > s3_6h[i] and close[i-1] >= r3_6h[i-1]:
                # Price crossed below R3 from above -> short fade
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            # Long fade at S3: price holds S3 and moves up toward PPT
            elif price > s3_6h[i] and price < r3_6h[i] and close[i-1] <= s3_6h[i-1]:
                # Price crossed above S3 from below -> long fade
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Breakout continuation at R4/S4: price closes beyond with volume
            elif price > r4_6h[i] and close[i-1] <= r4_6h[i-1]:
                # Price closed above R4 -> long breakout
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif price < s4_6h[i] and close[i-1] >= s4_6h[i-1]:
                # Price closed below S4 -> short breakout
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