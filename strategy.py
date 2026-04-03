#!/usr/bin/env python3
"""
Experiment #711: 6h Donchian Breakout + 1d Weekly Pivot + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts filtered by 1d weekly pivot direction (from 1d HTF) and volume confirmation
capture institutional breakout momentum with proper trend alignment. Works in bull/bear markets via pivot regime:
long only above weekly pivot, short only below weekly pivot. Target: 75-200 total trades over 4 years (19-50/year).
Uses discrete position sizing (0.25) to minimize fee churn and includes ATR-based stoploss.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_711_6h_donchian20_1d_pivot_vol_v1"
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
    
    # Calculate weekly pivot points from prior week (using daily OHLC)
    # For each day, calculate pivot based on prior week's high/low/close
    # We'll use a rolling window of 5 trading days (1 week) to get weekly OHLC
    lookback = 5  # 5 days = 1 week
    
    # Rolling weekly high, low, close (prior week's values)
    weekly_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    weekly_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    weekly_close = pd.Series(close_1d).rolling(window=lookback, min_periods=lookback).last().shift(1).values
    
    # Weekly Pivot Point = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly Support and Resistance levels
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    weekly_range = weekly_high - weekly_low
    r1 = 2 * weekly_pivot - weekly_low
    s1 = 2 * weekly_pivot - weekly_high
    r2 = weekly_pivot + weekly_range
    s2 = weekly_pivot - weekly_range
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 6h Indicators: Donchian Channel (20) ===
    lookback_dc = 20
    donchian_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    donchian_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(50, lookback_dc, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~36h on 6h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Get regime from weekly pivot: long above pivot, short below pivot
            # Only trade breakouts in direction of weekly pivot bias
            if price > donchian_high[i] and price > weekly_pivot_aligned[i]:
                # Bullish breakout above Donchian high and above weekly pivot
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donchian_low[i] and price < weekly_pivot_aligned[i]:
                # Bearish breakout below Donchian low and below weekly pivot
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals