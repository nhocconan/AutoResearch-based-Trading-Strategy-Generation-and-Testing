#!/usr/bin/env python3
"""
Experiment #2675: 6h Camarilla pivot fade/breakout with 1w trend filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with weekly trend filter and volume confirmation captures institutional 
order flow at key mathematical levels. Works in both bull/bear via regime adaptation.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2675_6h_camarilla_pivot_1w_trend_v1"
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
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3_1d = pivot_1d + range_1d * 1.1 / 2.0
    s3_1d = pivot_1d - range_1d * 1.1 / 2.0
    r4_1d = pivot_1d + range_1d * 1.1
    s4_1d = pivot_1d - range_1d * 1.1
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(21) for trend
    ema_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
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
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit conditions:
                # 1. Stop loss: 2*ATR below entry (using 6h range as proxy)
                atr_estimate = (high[i] - low[i]) * 0.15  # approximate ATR
                if price < entry_price - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # 2. Take profit at R4 for longs, or reverse signal
                elif price >= r4_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # 3. Reverse if price crosses S3 (mean reversion failed)
                elif price < s3_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                highest_since_entry = max(highest_since_entry, high[i])
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit conditions:
                # 1. Stop loss: 2*ATR above entry
                atr_estimate = (high[i] - low[i]) * 0.15
                if price > entry_price + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # 2. Take profit at S4 for shorts
                elif price <= s4_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # 3. Reverse if price crosses R3 (mean reversion failed)
                elif price > r3_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Weekly trend filter
        weekly_trend = trend_1w_aligned[i]
        
        # Volume confirmation: require moderate volume spike
        volume_ok = vol_ratio[i] > 1.3
        
        if volume_ok:
            # Long entry scenarios:
            # 1. Mean reversion long at S3 with weekly uptrend
            # 2. Breakout long above R4 with any weekly trend
            if ((weekly_trend > 0 and price <= s3_1d_aligned[i]) or
                price >= r4_1d_aligned[i]):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry scenarios:
            # 1. Mean reversion short at R3 with weekly downtrend
            # 2. Breakout short below S4 with any weekly trend
            elif ((weekly_trend < 0 and price >= r3_1d_aligned[i]) or
                  price <= s4_1d_aligned[i]):
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