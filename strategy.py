#!/usr/bin/env python3
"""
Experiment #5555: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 1.8x average and 
aligned with weekly trend (price above/below weekly pivot) capture high-probability 
trend moves. Weekly pivot provides robust support/resistance that adapts to longer-term 
volatility, while volume confirmation filters false breakouts. This combination should 
work in both bull (continuation breakouts) and bear (fade at weekly resistance) markets.
Target: 12-37 trades/year (50-150 total over 4 years) with discrete position sizing 
to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5555_6h_donchian20_1w_pivot_vol_v1"
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
    
    # === HTF: 1w data for weekly pivot levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 2:
        # Calculate weekly pivot levels from previous weekly bar
        # Standard pivot: P = (H + L + C) / 3
        # R1 = 2*P - L, S1 = 2*P - H
        # R2 = P + (H - L), S2 = P - (H - L)
        # Using previous bar's HLC to avoid look-ahead
        h_1w = df_1w['high'].values
        l_1w = df_1w['low'].values
        c_1w = df_1w['close'].values
        
        # Calculate pivot levels using previous bar (shifted by 1)
        pivot = (h_1w + l_1w + c_1w) / 3.0
        rng = h_1w - l_1w
        r1 = 2.0 * pivot - l_1w
        s1 = 2.0 * pivot - h_1w
        r2 = pivot + rng
        s2 = pivot - rng
        
        # Align to LTF (6h) with shift(1) for completed bars only
        pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
        r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
        r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
        s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    else:
        # Neutral values if insufficient data
        pivot_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14)  # Donchian, volume avg, ATR warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR Donchian lower band break OR price < S2 (weekly support)
                if price <= stop_price or price <= donchian_low[i] or price < s2_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR Donchian upper band break OR price > R2 (weekly resistance)
                if price >= stop_price or price >= donchian_high[i] or price > r2_aligned[i]:
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
        
        # Long: breakout above Donchian high with volume, above weekly pivot (trend) or below S1 (fade at support)
        long_entry = (breakout_up and volume_confirmed and 
                     (price > pivot_aligned[i] or price < s1_aligned[i]))
        # Short: breakout below Donchian low with volume, below weekly pivot (trend) or above R1 (fade at resistance)
        short_entry = (breakout_down and volume_confirmed and 
                      (price < pivot_aligned[i] or price > r1_aligned[i]))
        
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