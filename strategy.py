#!/usr/bin/env python3
"""
Experiment #2095: 6h Donchian(20) breakout + weekly Camarilla pivot + volume confirmation
HYPOTHESIS: Weekly Camarilla pivot levels (R3/S3, R4/S4) act as strong support/resistance on 6h timeframe.
Institutional order flow accumulates at these levels, causing breakouts with volume confirmation.
Only trade breakouts aligned with weekly trend (price above/below weekly VWAP).
Works in bull/bear markets by fading at R3/S3 (mean reversion) and breaking R4/S4 (continuation).
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2095_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Camarilla pivots and VWAP (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly VWAP
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vol_price_1w = typical_price_1w * volume_1w
    cum_vol_price = np.cumsum(vol_price_1w)
    cum_vol = np.cumsum(volume_1w)
    vwap_1w = np.divide(cum_vol_price, cum_vol, out=np.full_like(cum_vol_price, np.nan), where=cum_vol!=0)
    
    # Weekly trend: 1 if close > VWAP, -1 otherwise
    trend_1w = np.where(close_1w > vwap_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    # Camarilla: Pivot = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.5), R3 = C + ((H-L) * 1.25)
    # S3 = C - ((H-L) * 1.25), S4 = C - ((H-L) * 1.5)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r4_1w = close_1w + (range_1w * 1.5)
    r3_1w = close_1w + (range_1w * 1.25)
    s3_1w = close_1w - (range_1w * 1.25)
    s4_1w = close_1w - (range_1w * 1.5)
    
    # Align weekly levels to 6h (shifted by 1 for completed weekly bar only)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # === 6h Indicators: Donchian(20), Volume MA(20) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
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
    
    warmup = 60  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for stoploss
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops below S3 (mean reversion at support)
                if price <= s3_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches lower Donchian (trailing stop)
                elif price <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises above R3 (mean reversion at resistance)
                if price >= r3_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches upper Donchian (trailing stop)
                elif price >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require weekly trend alignment for bias filter
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Fade at R3/S3: mean reversion at extreme levels
            # Long: price rejects S3 and moves back above it with volume
            if trend_bias > 0 and price > s3_1w_aligned[i] and low[i] <= s3_1w_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: price rejects R3 and moves back below it with volume
            elif trend_bias < 0 and price < r3_1w_aligned[i] and high[i] >= r3_1w_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            # Breakout continuation at R4/S4: strong momentum
            # Long: price breaks above R4 with volume (bullish continuation)
            elif trend_bias > 0 and price > r4_1w_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: price breaks below S4 with volume (bearish continuation)
            elif trend_bias < 0 and price < s4_1w_aligned[i]:
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