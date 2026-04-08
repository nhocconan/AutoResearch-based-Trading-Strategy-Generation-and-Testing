#!/usr/bin/env python3
"""
Experiment #5891: 6h Donchian(20) breakout + 1d Camarilla pivot levels + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d Camarilla R4/S4 breakout levels (continuation) 
or R3/S3 fade levels (mean reversion in chop) capture high-probability moves. 
Volume confirmation filters weak breakouts. Works in bull/bear via dual regime logic.
Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5891_6h_donchian20_1d_camarilla_vol_v1"
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
        # Calculate Camarilla levels from previous day's OHLC
        # Camarilla: based on previous day's range
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_range = prev_high - prev_low
        
        # Camarilla levels
        r3 = prev_close + (prev_range * 1.1 / 4)
        r4 = prev_close + (prev_range * 1.1 / 2)
        s3 = prev_close - (prev_range * 1.1 / 4)
        s4 = prev_close - (prev_range * 1.1 / 2)
        
        # Align to LTF (6h) with shift(1) for completed bars only
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        r3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
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
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 20)  # Donchian, volume avg, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Camarilla-based logic:
        # Regime detection: if price > R4 or < S4 = strong breakout regime (continuation)
        # if price between R3-S3 = chop regime (mean reversion at R3/S3)
        strong_breakout_long = price > r4_aligned[i]
        strong_breakout_short = price < s4_aligned[i]
        chop_long = price < r3_aligned[i] and price > s3_aligned[i]  # inside R3-S3
        chop_short = chop_long  # same region
        
        # Entry conditions:
        # Strong breakout regime: continuation trades
        long_setup = breakout_up and volume_confirmed and strong_breakout_long
        short_setup = breakout_down and volume_confirmed and strong_breakout_short
        
        # Chop regime: mean reversion at R3/S3 levels
        long_setup_chop = breakout_down and volume_confirmed and chop_long and price <= s3_aligned[i]
        short_setup_chop = breakout_up and volume_confirmed and chop_short and price >= r3_aligned[i]
        
        if long_setup or long_setup_chop:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup or short_setup_chop:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals