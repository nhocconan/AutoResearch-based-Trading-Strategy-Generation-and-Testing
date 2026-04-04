#!/usr/bin/env python3
"""
Experiment #5671: 6h Donchian(20) breakout + 1d Camarilla pivot reversal + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 1.8x average and price 
near Camarilla S3/R3 levels from 1d timeframe capture high-probability reversal trades. 
Camarilla levels provide mathematical support/resistance that works in both bull and bear 
markets by identifying overextended moves. Volume confirms breakout strength at key levels. 
ATR trailing stop (2.5x) manages risk. Discrete sizing (0.25) minimizes fee churn. 
Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5671_6h_donchian20_1d_camarilla_vol_v1"
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
    
    # === HTF: 1d data for Camarilla pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate Camarilla pivot levels from prior day's OHLC
        # Camarilla formula: based on previous day's high, low, close
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3.0
        
        # Calculate range
        range_val = prev_high - prev_low
        
        # Camarilla levels
        # Resistance levels
        r1 = pivot + (range_val * 1.1 / 12)
        r2 = pivot + (range_val * 1.1 / 6)
        r3 = pivot + (range_val * 1.1 / 4)
        r4 = pivot + (range_val * 1.1 / 2)
        # Support levels
        s1 = pivot - (range_val * 1.1 / 12)
        s2 = pivot - (range_val * 1.1 / 6)
        s3 = pivot - (range_val * 1.1 / 4)
        s4 = pivot - (range_val * 1.1 / 2)
    else:
        # Fallback if insufficient data
        pivot = np.full(len(df_1d), np.nan)
        r1 = r2 = r3 = r4 = np.full(len(df_1d), np.nan)
        s1 = s2 = s3 = s4 = np.full(len(df_1d), np.nan)
    
    # Align Camarilla levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
    
    warmup = max(20, 20, 14, 1)  # Donchian, volume avg, ATR, Camarilla lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks above weekly R4 (strong bullish continuation - take profit)
                if price <= stop_price or price >= r4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks below weekly S4 (strong bearish continuation - take profit)
                if price >= stop_price or price <= s4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.8
        
        # Camarilla reversal logic: 
        # Long when price breaks above Donchian and is near S3 (strong support)
        # Short when price breaks below Donchian and is near R3 (strong resistance)
        near_s3 = price <= s3_aligned[i] * 1.02  # Within 2% above S3
        near_r3 = price >= r3_aligned[i] * 0.98  # Within 2% below R3
        
        # Entry conditions: Donchian breakout with volume confirmation near Camarilla S3/R3
        long_setup = breakout_up and volume_confirmed and near_s3
        short_setup = breakout_down and volume_confirmed and near_r3
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals