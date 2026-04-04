#!/usr/bin/env python3
"""
Experiment #5847: 6h Donchian(20) breakout + 1d/1w pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with daily/weekly pivot levels capture institutional order flow. 
In bull markets: breakouts above daily R1 with weekly bias long. In bear markets: breakdowns below daily S1 with weekly bias short. 
Volume confirmation filters weak breakouts. Pivot levels act as dynamic support/resistance that work in both regimes. 
Trailing stop manages risk. Targets 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5847_6h_donchian20_1d_1w_pivot_vol_v1"
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
    
    # === HTF: 1d data for daily pivot points ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate daily pivot points: P = (H+L+C)/3
        dp_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
        # R1 = 2*P - L, S1 = 2*P - H
        r1_1d = 2 * dp_1d - df_1d['low']
        s1_1d = 2 * dp_1d - df_1d['high']
        # Align to 6h timeframe
        dp_1d_aligned = align_htf_to_ltf(prices, df_1d, dp_1d.values)
        r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d.values)
        s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d.values)
    else:
        dp_1d_aligned = np.full(n, np.nan)
        r1_1d_aligned = np.full(n, np.nan)
        s1_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for weekly pivot bias ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Calculate weekly pivot points
        wp_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
        # Weekly bias: price above/below weekly pivot
        wp_1w_aligned = align_htf_to_ltf(prices, df_1w, wp_1w.values)
        weekly_bias_long = close > wp_1w_aligned  # bullish bias
        weekly_bias_short = close < wp_1w_aligned  # bearish bias
    else:
        wp_1w_aligned = np.full(n, np.nan)
        weekly_bias_long = np.zeros(n, dtype=bool)
        weekly_bias_short = np.zeros(n, dtype=bool)
    
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
    
    warmup = max(20, 20, 20, 14)  # Donchian, volume avg, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(dp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(wp_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below daily S1 (support loss)
                if price <= stop_price or price <= s1_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above daily R1 (resistance loss)
                if price >= stop_price or price >= r1_1d_aligned[i]:
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
        
        # Entry conditions: 
        # Long: breakout above Donchian high + above daily R1 + weekly bias long + volume
        # Short: breakdown below Donchian low + below daily S1 + weekly bias short + volume
        long_setup = breakout_up and (price > r1_1d_aligned[i]) and weekly_bias_long[i] and volume_confirmed
        short_setup = breakout_down and (price < s1_1d_aligned[i]) and weekly_bias_short[i] and volume_confirmed
        
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