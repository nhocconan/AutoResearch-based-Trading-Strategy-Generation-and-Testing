#!/usr/bin/env python3
"""
Experiment #6335: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: Tight Donchian breakouts on 6h with weekly Camarilla pivot bias (R3/S3 as key levels) 
and volume > 2.0x average capture institutional momentum. Weekly pivot provides structural bias 
that works in both bull (breakouts above R3 in uptrend) and bear (breakouts below S3 in downtrend) 
markets. Volume filter ensures breakouts have participation. Uses discrete sizing (0.25) to 
minimize fee churn. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6335_6h_donchian20_1w_pivot_vol_v1"
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
    if len(df_1w) >= 5:
        # Calculate weekly Camarilla pivot levels
        # Based on previous week's high, low, close
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        range_val = weekly_high - weekly_low
        
        # Camarilla levels: R3, R4, S3, S4
        r3 = pivot + range_val * 1.1 / 2.0
        r4 = pivot + range_val * 1.1
        s3 = pivot - range_val * 1.1 / 2.0
        s4 = pivot - range_val * 1.1
        
        # Align to 6h timeframe (using R3/S3 as bias levels)
        r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    else:
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14, 5) + 1  # Donchian, volume avg, ATR, weekly pivot + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks below Donchian low (failed breakout)
                # 3. Price crosses below weekly S3 (longer-term reversal)
                if price <= stop_price or price <= donchian_low[i] or price < s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks above Donchian high (failed breakout)
                # 3. Price crosses above weekly R3 (longer-term reversal)
                if price >= stop_price or price >= donchian_high[i] or price > r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0  # Strong volume filter
        
        # Entry logic: Donchian breakout with volume AND aligned with weekly Camarilla
        # LONG: breakout above Donchian high + volume + price > weekly R3
        # SHORT: breakout below Donchian low + volume + price < weekly S3
        long_entry = breakout_up and volume_confirmed and price > r3_aligned[i]
        short_entry = breakout_down and volume_confirmed and price < s3_aligned[i]
        
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