#!/usr/bin/env python3
"""
Experiment #4315: 6h Donchian(20) breakout + 1w Camarilla pivot + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h timeframe capture swing momentum when aligned with weekly Camarilla pivot levels (price between R3/S3 for mean reversion, breaks R4/S4 for continuation) and confirmed by volume (>1.5x average). Uses weekly pivots for structural support/resistance that works in both bull (continuation at R4/S4) and bear (reversal at R3/S3) markets. Targets 75-150 total trades over 4 years (19-37/year) to avoid fee drag. ATR trailing stop (2.0x) for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4315_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1w Camarilla pivot levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Calculate weekly Camarilla levels from previous week's OHLC
        # Camarilla: Close + (High-Low) * multipliers
        multipliers = [1.0/12, 1.0/6, 1.0/4, 1.0/2]  # for R1/S1, R2/S2, R3/S3, R4/S4
        h1w = df_1w['high'].values
        l1w = df_1w['low'].values
        c1w = df_1w['close'].values
        rng1w = h1w - l1w
        
        # Calculate R3, R4, S3, S4 levels
        r3 = c1w + rng1w * multipliers[2]  # 1/4
        r4 = c1w + rng1w * multipliers[3]  # 1/2
        s3 = c1w - rng1w * multipliers[2]  # 1/4
        s4 = c1w - rng1w * multipliers[3]  # 1/2
        
        # Align to 6h timeframe (shifted by 1 for completed weekly bar)
        r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
        r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
        s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    else:
        r3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
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
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Donchian breakout conditions (using previous bar's levels)
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # Camarilla pivot conditions
            in_middle_zone = (price > s3_aligned[i]) and (price < r3_aligned[i])  # Between S3/R3
            breakout_continuation = ((price > r4_aligned[i]) or (price < s4_aligned[i]))  # Beyond S4/R4
            
            # Long conditions: 
            # 1. Donchian breakout up AND price in middle zone (mean reversion long)
            # 2. OR Donchian breakout up AND price > R4 (continuation breakout)
            long_entry = (breakout_up and in_middle_zone) or (breakout_up and price > r4_aligned[i])
            
            # Short conditions:
            # 1. Donchian breakout down AND price in middle zone (mean reversion short)
            # 2. OR Donchian breakout down AND price < S4 (continuation breakdown)
            short_entry = (breakout_dn and in_middle_zone) or (breakout_dn and price < s4_aligned[i])
            
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