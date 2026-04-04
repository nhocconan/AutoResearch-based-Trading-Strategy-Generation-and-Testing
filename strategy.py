#!/usr/bin/env python3
"""
Experiment #3011: 6h Camarilla Pivot Fade/Breakout + 1d Volume Confirmation
HYPOTHESIS: Camarilla pivot levels from daily timeframe act as strong support/resistance.
At R3/S3 levels, price tends to fade (mean revert) in ranging markets. At R4/S4 levels,
price tends to breakout with continuation in trending markets. Volume confirmation (>1.5x
20-period average) filters false signals. 6h timeframe balances trade frequency and
captures multi-day swings. Works in both bull/bear via adaptive logic: fade at inner
levels, breakout at outer levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3011_6h_camarilla_pivot_fade_breakout_vol_v1"
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
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # R3 = PP + (H - L) * 1.1/4
    # S3 = PP - (H - L) * 1.1/4
    # S4 = PP - (H - L) * 1.1/2
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r4_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2.0
    r3_1d = pp_1d + (high_1d - low_1d) * 1.1 / 4.0
    s3_1d = pp_1d - (high_1d - low_1d) * 1.1 / 4.0
    s4_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align 1d levels to 6h timeframe (with shift(1) for completed bars only)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
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
    
    warmup = 50  # sufficient for volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
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
                # Exit if price reaches opposite Camarilla level (mean reversion or breakout failed)
                elif price <= s3_1d_aligned[i]:  # Reached S3, exit long
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
                # Exit if price reaches opposite Camarilla level (mean reversion or breakout failed)
                elif price >= r3_1d_aligned[i]:  # Reached R3, exit short
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) for confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Fade logic at R3/S3 (mean reversion in ranging markets)
            # Short near R3, Long near S3
            if price >= r3_1d_aligned[i] and price < r4_1d_aligned[i]:
                # In R3-R4 zone: potential short on rejection from R3
                if price < (r3_1d_aligned[i] + r4_1d_aligned[i]) / 2:  # Below midpoint of R3-R4
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
            elif price <= s3_1d_aligned[i] and price > s4_1d_aligned[i]:
                # In S3-S4 zone: potential long on rejection from S3
                if price > (s3_1d_aligned[i] + s4_1d_aligned[i]) / 2:  # Above midpoint of S3-S4
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
            # Breakout logic at R4/S4 (continuation in trending markets)
            elif price > r4_1d_aligned[i]:
                # Break above R4: go long
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif price < s4_1d_aligned[i]:
                # Break below S4: go short
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