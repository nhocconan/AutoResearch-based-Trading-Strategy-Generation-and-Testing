#!/usr/bin/env python3
"""
Experiment #6379: 6h Donchian(20) breakout + 12h Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts with volume confirmation (>1.8x avg) and 12h Camarilla pivot direction (price above/below pivot) capture institutional momentum while avoiding whipsaws. Camarilla pivot provides stronger trend bias than EMA: price above daily pivot = bullish bias, below = bearish. Volume confirmation filters false breakouts. Discrete sizing (0.25) minimizes fee churn. Target: 75-200 trades over 4 years. Works in bull via breakouts, in bear via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6379_6h_donchian20_12h_camarilla_vol_v1"
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
    
    # === HTF: 12h data for Camarilla pivot ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 2:
        # Calculate Camarilla pivot from previous 12h bar
        prev_high = df_12h['high'].shift(1).values
        prev_low = df_12h['low'].shift(1).values
        prev_close = df_12h['close'].shift(1).values
        pivot = (prev_high + prev_low + prev_close) / 3.0
        camarilla_high = pivot + 1.1 * (prev_high - prev_low) / 2.0  # R4 level
        camarilla_low = pivot - 1.1 * (prev_high - prev_low) / 2.0   # S4 level
        # Align to 6h timeframe
        pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
        camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
        camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    else:
        pivot_aligned = np.full(n, np.nan)
        camarilla_high_aligned = np.full(n, np.nan)
        camarilla_low_aligned = np.full(n, np.nan)
    
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
            np.isnan(pivot_aligned[i]) or np.isnan(camarilla_high_aligned[i]) or
            np.isnan(camarilla_low_aligned[i])):
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
                # 3. Price crosses below 12h Camarilla low (trend change)
                if price <= stop_price or price <= donchian_low[i] or price < camarilla_low_aligned[i]:
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
                # 3. Price crosses above 12h Camarilla high (trend change)
                if price >= stop_price or price >= donchian_high[i] or price > camarilla_high_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.8  # Volume filter
        
        # Entry logic based on 12h Camarilla pivot trend filter:
        # Long: breakout up + volume + price > 12h pivot (bullish bias)
        # Short: breakout down + volume + price < 12h pivot (bearish bias)
        
        long_entry = breakout_up and volume_confirmed and (price > pivot_aligned[i])
        short_entry = breakout_down and volume_confirmed and (price < pivot_aligned[i])
        
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

</think>