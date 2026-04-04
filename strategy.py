#!/usr/bin/env python3
"""
Experiment #3511: 6h Camarilla Pivot Breakout + 1d Volume Spike + ATR Filter
HYPOTHESIS: 6h price breaking Camarilla R4/S4 levels from 1d data with 1d volume confirmation captures institutional breakout moves. 
Camarilla pivots (R4/S4) represent extreme intraday levels where breakouts often accelerate. Volume confirms institutional participation. 
ATR-based trailing stop manages risk. Works in bull (R4 breakouts) and bear (S4 breakdowns) via symmetric logic.
Position size 0.25. Target: 60-150 total trades over 4 years (15-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3511_6h_camarilla_breakout_1d_vol_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and volume (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for each 1d bar using prior day's OHLC
    # Camarilla formulas: based on previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point (not used in Camarilla but needed for calculations)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r4 = close_1d + range_ * 1.1 / 2  # R4 = Close + (High-Low)*1.1/2
    r3 = close_1d + range_ * 1.1 / 4  # R3 = Close + (High-Low)*1.1/4
    s3 = close_1d - range_ * 1.1 / 4  # S3 = Close - (High-Low)*1.1/4
    s4 = close_1d - range_ * 1.1 / 2  # S4 = Close - (High-Low)*1.1/2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 1d Volume MA(20) for spike detection ===
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones(len(vol_1d))
    vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 14)  # sufficient for ATR and Camarilla (need prior day)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below S3 (mean reversion toward pivot)
                elif price < s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above R3 (mean reversion toward pivot)
                elif price > r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        if volume_spike:
            # Long entry: price breaks above R4 (extreme bullish breakout)
            if price > r4_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below S4 (extreme bearish breakdown)
            elif price < s4_aligned[i]:
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