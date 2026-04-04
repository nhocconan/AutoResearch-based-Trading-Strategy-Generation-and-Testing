#!/usr/bin/env python3
"""
Experiment #3891: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d Camarilla pivot levels capture institutional order flow.
Breakouts at R4/S4 levels with volume > 1.8x MA(30) indicate strong momentum. Fade at R3/S3 with volume < 0.7x MA.
Uses ATR(14) trailing stop (2.0x) for risk management. Works in both bull and bear markets by adapting to pivot levels.
Target: 100-200 trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3891_6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla pivots from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    # Camarilla levels
    r3 = pivot + range_hl * 1.1 / 2
    s3 = pivot - range_hl * 1.1 / 2
    r4 = pivot + range_hl * 1.1
    s4 = pivot - range_hl * 1.1
    # Align to 6h timeframe (shifted by 1 for completed bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(30) for spike/dry-up detection ===
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[30:] = volume[30:] / vol_ma[30:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 30)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
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
                if price > lowest_since_entry + 2.0 * atr[i]:
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
        # Volume conditions: spike for breakout, dry-up for fade
        volume_spike = vol_ratio[i] > 1.8
        volume_dry = vol_ratio[i] < 0.7
        
        # Long entry: breakout above Donchian upper band at R4 level with volume spike
        long_breakout_r4 = price > highest_high[i-1] and price > r4_aligned[i] and volume_spike
        # Long entry: fade at S3 level with volume dry-up (mean reversion)
        long_fade_s3 = price < lowest_low[i-1] and price < s3_aligned[i] and volume_dry and price > s4_aligned[i]
        
        # Short entry: breakdown below Donchian lower band at S4 level with volume spike
        short_breakout_s4 = price < lowest_low[i-1] and price < s4_aligned[i] and volume_spike
        # Short entry: fade at R3 level with volume dry-up (mean reversion)
        short_fade_r3 = price > highest_high[i-1] and price > r3_aligned[i] and volume_dry and price < r4_aligned[i]
        
        if long_breakout_r4 and not (short_breakout_s4 or short_fade_r3):
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif long_fade_s3 and not (short_breakout_s4 or short_fade_r3):
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_breakout_s4 and not (long_breakout_r4 or long_fade_s3):
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        elif short_fade_r3 and not (long_breakout_r4 or long_fade_s3):
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals