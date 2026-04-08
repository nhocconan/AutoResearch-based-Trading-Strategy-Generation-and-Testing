#!/usr/bin/env python3
"""
Experiment #4119: 6h Donchian(20) breakout + 12h Camarilla pivot levels + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
and volume confirmation capture institutional order flow. Camarilla levels act as natural support/resistance - 
take mean reversion trades at R3/S3 and breakout continuation at R4/S4. Works in bull/bear by using price 
action relative to pivot levels. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4119_6h_donchian20_12h_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h Camarilla pivot levels ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 2:
        # Calculate Camarilla levels for each 12h bar
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # Camarilla formula: range = high - low
        # R4 = close + range * 1.1/2
        # R3 = close + range * 1.1/4
        # S3 = close - range * 1.1/4
        # S4 = close - range * 1.1/2
        range_12h = high_12h - low_12h
        r4_12h = close_12h + range_12h * 1.1 / 2
        r3_12h = close_12h + range_12h * 1.1 / 4
        s3_12h = close_12h - range_12h * 1.1 / 4
        s4_12h = close_12h - range_12h * 1.1 / 2
        
        # Align to 6h timeframe (shifted by 1 for completed bars only)
        r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
        r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
        s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
        s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    else:
        r4_12h_aligned = np.full(n, np.nan)
        r3_12h_aligned = np.full(n, np.nan)
        s3_12h_aligned = np.full(n, np.nan)
        s4_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(20) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 10)  # DC lookback, vol MA buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i])):
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
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Donchian breakout logic
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Camarilla level conditions
            # Mean reversion: price at R3/S3 levels
            # Breakout continuation: price breaks R4/S4 levels
            
            # Long conditions:
            # 1. Mean reversion long: price near S3 and bouncing up
            long_mean_rev = (price <= s3_12h_aligned[i] * 1.002) and (price > low[i]) and breakout_up
            # 2. Breakout continuation long: price breaks above R4
            long_breakout = price > r4_12h_aligned[i] and breakout_up
            
            # Short conditions:
            # 1. Mean reversion short: price near R3 and bouncing down
            short_mean_rev = (price >= r3_12h_aligned[i] * 0.998) and (price < high[i]) and breakout_down
            # 2. Breakout continuation short: price breaks below S4
            short_breakout = price < s4_12h_aligned[i] and breakout_down
            
            if long_mean_rev or long_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_mean_rev or short_breakout:
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