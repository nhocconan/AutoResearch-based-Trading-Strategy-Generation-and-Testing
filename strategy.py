#!/usr/bin/env python3
"""
Experiment #4207: 6h Donchian(20) breakout + 1d pivot direction + volume confirmation
HYPOTHESIS: Donchian channel breakouts on 6h timeframe capture momentum when aligned with 1d Camarilla pivot bias
(R3/S3 for mean reversion, R4/S4 for breakout) and confirmed by volume (>1.8x average). The 1d pivot filter ensures
we trade with the higher timeframe structure, avoiding false breakouts in ranging markets. Discrete position sizing
(0.25) limits fee churn, targeting 75-150 total trades over 4 years (19-38/year). Works in both bull and bear markets
by using pivot levels as dynamic support/resistance that adapt to volatility regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4207_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Precompute HTF: 1d data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Camarilla pivot calculation using previous day's OHLC
        # We need to shift by 1 to avoid look-ahead (use previous completed day)
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_hl = prev_high - prev_low
        
        # Camarilla levels
        r3 = pivot + (range_hl * 1.1 / 4.0)
        s3 = pivot - (range_hl * 1.1 / 4.0)
        r4 = pivot + (range_hl * 1.1 / 2.0)
        s4 = pivot - (range_hl * 1.1 / 2.0)
        
        # Align to 6h timeframe (shift(1) already applied above for no look-ahead)
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.8x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.8
        
        if volume_confirm:
            # Donchian breakout conditions
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # Determine market regime based on 1d pivot position
            # Near S3/R3: mean reversion zone (fade extremes)
            # Beyond S4/R4: breakout zone (continuation)
            near_support = price <= s3_aligned[i] * 1.02 and price >= s3_aligned[i] * 0.98
            near_resistance = price <= r3_aligned[i] * 1.02 and price >= r3_aligned[i] * 0.98
            breakout_support = price < s4_aligned[i]
            breakout_resistance = price > r4_aligned[i]
            
            # Long conditions:
            # 1. Mean reversion: Donchian breakout up near S3 (potential bounce)
            # 2. Breakout: Donchian breakout up beyond R4 (continuation)
            long_mean_rev = breakout_up and near_support
            long_breakout = breakout_up and breakout_resistance
            long_entry = long_mean_rev or long_breakout
            
            # Short conditions:
            # 1. Mean reversion: Donchian breakout down near R3 (potential rejection)
            # 2. Breakout: Donchian breakout down beyond S4 (continuation)
            short_mean_rev = breakout_dn and near_resistance
            short_breakout = breakout_dn and breakout_support
            short_entry = short_mean_rev or short_breakout
            
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
        else:
            signals[i] = 0.0
    
    return signals