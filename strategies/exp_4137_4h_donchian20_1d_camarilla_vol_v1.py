#!/usr/bin/env python3
"""
Experiment #4137: 4h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: 4h Donchian breakouts aligned with 1d Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) capture institutional order flow. Volume confirmation filters false breakouts. Works in both bull/bear as Camarilla adapts to volatility. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4137_4h_donchian20_1d_camarilla_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        daily_range = high_1d - low_1d
        # Camarilla levels: H3/L3 = close ± 1.1*range/2, H4/L4 = close ± 1.1*range
        camarilla_h3 = close_1d + 1.1 * daily_range / 2
        camarilla_l3 = close_1d - 1.1 * daily_range / 2
        camarilla_h4 = close_1d + 1.1 * daily_range
        camarilla_l4 = close_1d - 1.1 * daily_range
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    else:
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for volatility and stoploss ===
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
    
    warmup = max(lookback_dc + 1, 20 + 5, 14 + 5)  # DC lookback, vol MA buffer, ATR buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) to filter noise
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Donchian breakout logic
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Camarilla pivot logic
            at_h3 = abs(price - camarilla_h3_aligned[i]) < 0.1 * camarilla_h3_aligned[i]
            at_l3 = abs(price - camarilla_l3_aligned[i]) < 0.1 * camarilla_l3_aligned[i]
            above_h4 = price > camarilla_h4_aligned[i]
            below_l4 = price < camarilla_l4_aligned[i]
            
            # Long conditions: Donchian breakout up + above H4 (continuation) OR at L3 (mean reversion)
            long_entry = (breakout_up and above_h4) or (breakout_up and at_l3)
            
            # Short conditions: Donchian breakout down + below L4 (continuation) OR at H3 (mean reversion)
            short_entry = (breakout_down and below_l4) or (breakout_down and at_h3)
            
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
        else:
            signals[i] = 0.0
    
    return signals