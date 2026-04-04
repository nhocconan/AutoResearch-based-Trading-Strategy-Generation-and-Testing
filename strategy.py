#!/usr/bin/env python3
"""
Experiment #5951: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h aligned with 1d Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) capture institutional interest.
Volume >1.5x average confirms participation. ATR trailing stop manages risk. Target: 75-200 trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5951_6h_donchian20_1d_camarilla_vol_v1"
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
        # Calculate Camarilla levels from prior day's OHLC
        high_1d = pd.Series(df_1d['high'].values)
        low_1d = pd.Series(df_1d['low'].values)
        close_1d = pd.Series(df_1d['close'].values)
        
        # Prior day's values (shifted by 1)
        prev_high = high_1d.shift(1).values
        prev_low = low_1d.shift(1).values
        prev_close = close_1d.shift(1).values
        
        # Camarilla levels
        range_ = prev_high - prev_low
        camarilla_h4 = prev_close + range_ * 1.1 / 2
        camarilla_l4 = prev_close - range_ * 1.1 / 2
        camarilla_h3 = prev_close + range_ * 1.1 / 4
        camarilla_l3 = prev_close - range_ * 1.1 / 4
        camarilla_h2 = prev_close + range_ * 1.1 / 6
        camarilla_l2 = prev_close - range_ * 1.1 / 6
        camarilla_h1 = prev_close + range_ * 1.1 / 12
        camarilla_l1 = prev_close - range_ * 1.1 / 12
        
        # Align all levels to LTF
        h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    else:
        h4_aligned = l4_aligned = h3_aligned = l3_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14, 2) + 1  # Donchian, volume avg, ATR, prior day + shift
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                # OR price reaches L3 (take profit at 75% of range)
                if price <= stop_price or price <= donchian_low[i] or price <= l3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                # OR price reaches H3 (take profit at 75% of range)
                if price >= stop_price or price >= donchian_high[i] or price >= h3_aligned[i]:
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
        
        # Camarilla filters:
        # Long: breakout above H4 (strong breakout) OR bounce from L3 (mean reversion in range)
        # Short: breakout below L4 (strong breakdown) OR bounce from H3 (mean reversion in range)
        long_breakout = breakout_up and volume_confirmed and price > h4_aligned[i]
        long_reversion = (price > l3_aligned[i] and price < l4_aligned[i]) and volume_confirmed and price > close[i-1]
        long_setup = long_breakout or long_reversion
        
        short_breakout = breakout_down and volume_confirmed and price < l4_aligned[i]
        short_reversion = (price < h3_aligned[i] and price > h4_aligned[i]) and volume_confirmed and price < close[i-1]
        short_setup = short_breakout or short_reversion
        
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