#!/usr/bin/env python3
"""
Experiment #2687: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot bias (from 1d HTF) and volume spikes
capture institutional participation with lower frequency than 4h strategies. Weekly pivots provide
structure-aware directional filter that works in both bull/bear regimes by identifying key
supply/demand levels. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2687_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Since we have daily data, we approximate weekly by taking prior 5-day period
    # But simpler: use prior day's range for intraday pivots (more responsive)
    # Standard floor pivot: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    
    # Calculate pivots for each day using prior day's data (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first day has no prior
    
    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    pivot_range = prev_high - prev_low
    
    # R3 and S3 levels (strong support/resistance)
    r3 = prev_high + 2.0 * (pivot_point - prev_low)
    s3 = prev_low - 2.0 * (prev_high - pivot_point)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Determine bias: above pivot = bullish bias, below = bearish bias
    # But stronger signal when near R3/S3 for reversals or breaks for continuation
    # We'll use: bias = 1 if close > pivot, -1 if close < pivot
    bias_raw = np.where(close_1d > pivot_point, 1, -1)
    bias_aligned = align_htf_to_ltf(prices, df_1d, bias_raw)
    
    # === 6h Indicators: Donchian(20) channels, Volume MA(20) ===
    # Donchian channels (20-period high/low)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(bias_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry (using Donchian width as ATR proxy)
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15  # approximate ATR from channel width
                if price < highest_since_entry - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian low (mean reversion)
                elif price < lowest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15
                if price > lowest_since_entry + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian high (mean reversion)
                elif price > highest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine pivot proximity and bias
            near_r3 = abs(price - r3_aligned[i]) / price < 0.005  # within 0.5%
            near_s3 = abs(price - s3_aligned[i]) / price < 0.005
            above_pivot = price > pivot_aligned[i]
            below_pivot = price < pivot_aligned[i]
            
            # Long logic:
            # 1. Continuation: price breaks above R3 with bullish bias
            # 2. Reversal: price rejects S3 with bullish bias and breaks above pivot
            if (near_r3 and bias_aligned[i] > 0 and price > r3_aligned[i]) or \
               (near_s3 and bias_aligned[i] > 0 and price > pivot_aligned[i] and price > highest_20[i]):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short logic:
            # 1. Continuation: price breaks below S3 with bearish bias
            # 2. Reversal: price rejects R3 with bearish bias and breaks below pivot
            elif (near_s3 and bias_aligned[i] < 0 and price < s3_aligned[i]) or \
                 (near_r3 and bias_aligned[i] < 0 and price < pivot_aligned[i] and price < lowest_20[i]):
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