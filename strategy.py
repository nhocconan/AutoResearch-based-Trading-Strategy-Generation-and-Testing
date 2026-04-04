#!/usr/bin/env python3
"""
Experiment #4275: 6h Donchian(20) breakout + 1w Camarilla pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture swing momentum when aligned with weekly Camarilla pivot structure (price > weekly R3 for longs, < weekly S3 for shorts) and confirmed by volume (>2.0x average). Weekly pivots provide institutional support/resistance levels that work in both bull (breakout continuation) and bear (fade at R4/S4) markets. Position size 0.25 targets 75-150 total trades over 4 years (19-37/year). Uses 1w HTF as specified in experiment to reduce noise while maintaining sufficient trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4275_6h_donchian20_1w_camarilla_vol_v1"
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
    
    # === Precompute HTF: 1w Camarilla Pivot Levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 2:
        # Calculate weekly Camarilla pivot levels from previous week
        # PP = (H + L + C) / 3
        # R4 = PP + (H - L) * 1.1/2
        # R3 = PP + (H - L) * 1.1/4
        # R2 = PP + (H - L) * 1.1/6
        # R1 = PP + (H - L) * 1.1/12
        # S1 = PP - (H - L) * 1.1/12
        # S2 = PP - (H - L) * 1.1/6
        # S3 = PP - (H - L) * 1.1/4
        # S4 = PP - (H - L) * 1.1/2
        
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        pp = (weekly_high + weekly_low + weekly_close) / 3.0
        rng = weekly_high - weekly_low
        
        r4 = pp + rng * 1.1 / 2.0
        r3 = pp + rng * 1.1 / 4.0
        s3 = pp - rng * 1.1 / 4.0
        s4 = pp - rng * 1.1 / 2.0
        
        # Align to 6h timeframe (use previous week's levels)
        r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    else:
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
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
            np.isnan(atr[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
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
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        if volume_confirm:
            # Donchian breakout conditions (using previous bar's levels)
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # Weekly Camarilla pivot filter
            price_above_r3 = price > r3_aligned[i]
            price_below_s3 = price < s3_aligned[i]
            
            # Long conditions: Donchian breakout up + price above weekly R3
            long_entry = breakout_up and price_above_r3
            
            # Short conditions: Donchian breakout down + price below weekly S3
            short_entry = breakout_dn and price_below_s3
            
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