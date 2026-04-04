#!/usr/bin/env python3
"""
Experiment #4227: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture swing momentum when aligned with 1d Camarilla pivot levels (R3/S3 for fade, R4/S4 for continuation) and confirmed by volume (>1.8x average). Uses 1d pivot for structure, 6h for timing. ATR trailing stop (2.0x) manages risk. Position size 0.25 targets 50-150 total trades over 4 years (12-37/year). Works in bull via breakout continuation at R4/S4, in bear via fading at R3/S3 or shorting breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4227_6h_donchian20_1d_camarilla_vol_v1"
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
    
    # === Precompute HTF: 1d OHLC for Camarilla pivot ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
        pivot = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
        range_hl = df_1d['high'].values - df_1d['low'].values
        r3 = pivot + 1.1 * range_hl / 2
        s3 = pivot - 1.1 * range_hl / 2
        r4 = pivot + 1.1 * range_hl
        s4 = pivot - 1.1 * range_hl
        # Align to 6h timeframe
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pivot_aligned = np.full(n, np.nan)
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
            np.isnan(atr[i]) or np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
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
        # Require volume confirmation (> 1.8x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.8
        
        if volume_confirm:
            # Donchian breakout conditions (using previous bar's levels)
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # Price relative to Camarilla levels
            price_above_r4 = price > r4_aligned[i]
            price_below_s4 = price < s4_aligned[i]
            price_between_r3_s4 = (price > r3_aligned[i]) and (price < r4_aligned[i])
            price_between_s3_r4 = (price > s3_aligned[i]) and (price < s4_aligned[i])
            price_below_s3 = price < s3_aligned[i]
            price_above_r3 = price > r3_aligned[i]
            
            # Long conditions:
            # 1. Breakout above R4 (strong continuation)
            # 2. Fade from S3 (mean reversion in lower range)
            long_continuation = breakout_up and price_above_r4
            long_fade = breakout_up and price_between_s3_r4  # Pullback to S3 area then break up
            
            # Short conditions:
            # 1. Breakout below S4 (strong continuation)
            # 2. Fade from R3 (mean reversion in upper range)
            short_continuation = breakout_dn and price_below_s4
            short_fade = breakout_dn and price_between_r3_s4  # Pullback to R3 area then break down
            
            if long_continuation or long_fade:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_continuation or short_fade:
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