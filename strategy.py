#!/usr/bin/env python3
"""
Experiment #6255: 6h Donchian(20) breakout + 1w Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1-week Camarilla pivot levels (R4/S4 for continuation, R3/S3 for mean reversion) capture institutional order flow. Volume >2.0x average confirms participation. Uses 1w HTF for pivot calculation (proven effective for identifying key supply/demand zones). Discrete sizing (0.25) manages fee drag. Target: 75-200 trades over 4 years (19-50/year) for 6h timeframe. Works in both bull (breakout continuation) and bear (mean reversion at extremes) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6255_6h_donchian20_1w_camarilla_vol_v1"
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
    if len(df_1w) >= 2:  # Need at least 2 weekly bars for pivot calculation
        # Calculate Camarilla pivot levels from previous week
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # Camarilla pivot formula (based on previous week's range)
        pivot = (high_1w + low_1w + close_1w) / 3
        range_1w = high_1w - low_1w
        
        # Camarilla levels
        r4 = pivot + (range_1w * 1.1 / 2)
        r3 = pivot + (range_1w * 1.1 / 4)
        s3 = pivot - (range_1w * 1.1 / 4)
        s4 = pivot - (range_1w * 1.1 / 2)
        
        # Align to 6h timeframe (shift(1) inside align_htf_to_ltf for completed bars only)
        r4_6h = align_htf_to_ltf(prices, df_1w, r4)
        r3_6h = align_htf_to_ltf(prices, df_1w, r3)
        s3_6h = align_htf_to_ltf(prices, df_1w, s3)
        s4_6h = align_htf_to_ltf(prices, df_1w, s4)
    else:
        r4_6h = np.full(n, np.nan)
        r3_6h = np.full(n, np.nan)
        s3_6h = np.full(n, np.nan)
        s4_6h = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14) + 1  # Donchian, volume avg, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i])):
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
                # 3. Mean reversion: price reaches S3 (strong support) in bullish context
                if price <= stop_price or price <= donchian_low[i] or price <= s3_6h[i]:
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
                # 3. Mean reversion: price reaches R3 (strong resistance) in bearish context
                if price >= stop_price or price >= donchian_high[i] or price >= r3_6h[i]:
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
        
        # Entry logic based on Camarilla zones:
        # LONG: 
        #   - Breakout above Donchian high with volume AND price > R4 (continuation)
        #   - OR mean reversion from extreme low: price < S4 AND breaking above Donchian low with volume
        # SHORT:
        #   - Breakout below Donchian low with volume AND price < S4 (continuation)
        #   - OR mean reversion from extreme high: price > R4 AND breaking below Donchian high with volume
        
        long_breakout = breakout_up and volume_confirmed and price > r4_6h[i]
        long_mean_reversion = (price < s4_6h[i]) and breakout_up and volume_confirmed and price > donchian_low[i-1]
        
        short_breakout = breakout_down and volume_confirmed and price < s4_6h[i]
        short_mean_reversion = (price > r4_6h[i]) and breakout_down and volume_confirmed and price < donchian_high[i-1]
        
        long_entry = long_breakout or long_mean_reversion
        short_entry = short_breakout or short_mean_reversion
        
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