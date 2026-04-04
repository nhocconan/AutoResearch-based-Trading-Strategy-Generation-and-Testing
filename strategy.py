#!/usr/bin/env python3
"""
Experiment #5535: 6h Donchian(20) breakout + 1w Camarilla pivot + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 1.3x average and aligned with 
1w Camarilla pivot structure (fade at R3/S3, breakout at R4/S4) capture institutional-level 
moves that persist across bull and bear markets. Weekly pivots provide stronger structural 
support/resistance than daily, reducing false breakouts. Volume confirmation filters low-
conviction moves. Discrete position sizing (0.25) and ATR-based trailing stop control risk. 
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5535_6h_donchian20_1w_camarilla_vol_v1"
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
    
    # === HTF: 1w data for Camarilla pivot levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 2:
        # Calculate Camarilla pivot levels from previous week's OHLC
        prev_close = df_1w['close'].shift(1).values
        prev_high = df_1w['high'].shift(1).values
        prev_low = df_1w['low'].shift(1).values
        
        # Pivot point (PP) = (H + L + C) / 3
        pp = (prev_high + prev_low + prev_close) / 3.0
        # Range = H - L
        rang = prev_high - prev_low
        
        # Camarilla levels:
        r4 = pp + rang * 1.1 / 2.0
        r3 = pp + rang * 1.1 / 4.0
        s3 = pp - rang * 1.1 / 4.0
        s4 = pp - rang * 1.1 / 2.0
        
        # Align to LTF (6h) with shift(1) for completed bars only
        r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
        r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    else:
        r4_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 20, 14)
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (21-23 UTC) ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR broken breakout OR pivot failure
                if price <= stop_price or price <= donchian_low[i] or price < s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR broken breakout OR pivot failure
                if price >= stop_price or price >= donchian_high[i] or price > r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions (using previous bar's bands)
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume_ratio[i] > 1.3
        
        # Camarilla-based entry logic:
        # Long: strong breakout above R4 OR bounce from S3 (mean reversion)
        # Short: strong breakdown below S4 OR bounce from R3 (mean reversion)
        long_breakout = breakout_up and price > r4_aligned[i-1]
        long_bounce = price > s3_aligned[i] and low[i] <= s3_aligned[i]
        short_breakout = breakout_down and price < s4_aligned[i-1]
        short_bounce = price < r3_aligned[i] and high[i] >= r3_aligned[i]
        
        # Enter on breakout/bounce with volume confirmation
        if (long_breakout or long_bounce) and volume_confirmed:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif (short_breakout or short_bounce) and volume_confirmed:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals