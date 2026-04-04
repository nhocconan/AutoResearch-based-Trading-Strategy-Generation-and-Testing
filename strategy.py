#!/usr/bin/env python3
"""
Experiment #6111: 6h Camarilla Pivot Breakout + 1d Volume Spike + ATR Trailing Stop
HYPOTHESIS: 6h price breaks above/below Camarilla R4/S4 levels from prior 1d with volume >2x average capture institutional breakout moves. 
Camarilla levels derived from 1d OHLC provide mathematically precise support/resistance that works in both bull/bear markets. 
Volume confirmation ensures breakout legitimacy. ATR(14) trailing stop (2.5x) manages risk. 
Discrete sizing (0.25) minimizes fee churn. Target: 75-150 trades over 4 years.
Timeframe: 6h. HTF: 1d for Camarilla calculation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6111_6h_camarilla_breakout_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Calculate Camarilla levels from prior 1d OHLC
        h1 = df_1d['high'].values
        l1 = df_1d['low'].values
        c1 = df_1d['close'].values
        
        # Camarilla levels: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
        camarilla_r4 = c1 + ((h1 - l1) * 1.1 / 2)
        camarilla_s4 = c1 - ((h1 - l1) * 1.1 / 2)
        
        # Align to 6h timeframe (shifted by 1 for completed 1d bar only)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14) + 1  # volume avg, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks back below R3 (failed breakout)
                camarilla_r3 = camarilla_s4_aligned[i] + 4 * (camarilla_r4_aligned[i] - camarilla_s4_aligned[i]) / 6
                if price <= stop_price or price <= camarilla_r3:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks back above S3 (failed breakout)
                camarilla_s3 = camarilla_r4_aligned[i] - 4 * (camarilla_r4_aligned[i] - camarilla_s4_aligned[i]) / 6
                if price >= stop_price or price >= camarilla_s3:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > camarilla_r4_aligned[i]
        breakout_down = price < camarilla_s4_aligned[i]
        volume_confirmed = volume_ratio[i] > 2.0  # Volume filter for stronger signals
        
        # Entry conditions:
        # Long: breakout above R4 with volume confirmation
        # Short: breakout below S4 with volume confirmation
        long_entry = breakout_up and volume_confirmed
        short_entry = breakout_down and volume_confirmed
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals