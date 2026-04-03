#!/usr/bin/env python3
"""
Experiment #218: 1d Donchian Breakout + 1w HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian(20) breakouts on 1d timeframe capture medium-term trends.
1w HMA(21) acts as a trend filter - only take breakouts in direction of weekly trend.
Volume confirmation (>1.5x average) ensures breakout conviction.
ATR-based stoploss (2.5x) manages risk.
Designed for 1d timeframe targeting 7-25 trades/year (30-100 total over 4 years).
Works in both bull (breakouts with trend) and bear (failed reversals, short breakdowns) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_218_1d_donchian_1w_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on weekly close
    def calculate_hma(arr, period):
        """Hull Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = arr.rolling(window=period//2, min_periods=period//2).mean()
        full = arr.rolling(window=period, min_periods=period).mean()
        raw_hma = 2.0 * half - full
        hma = raw_hma.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
        return hma.values
    
    hma_21_1w = calculate_hma(pd.Series(df_1w['close'].values), 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: 1w HMA direction ---
        # Calculate HMA slope over 3 periods to determine trend
        if i >= 3:
            hma_now = hma_21_1w_aligned[i]
            hma_prev = hma_21_1w_aligned[i-3]
            hma_slope = hma_now - hma_prev
            trend_up = hma_slope > 0
            trend_down = hma_slope < 0
        else:
            trend_up = False
            trend_down = False
        
        # --- Volume Confirmation ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Price Levels ---
        price = close[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 days to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long breakout: Price > upper Donchian + volume spike + weekly trend up
        long_breakout = (price > upper_channel) and volume_spike and trend_up
        
        # Short breakdown: Price < lower Donchian + volume spike + weekly trend down
        short_breakout = (price < lower_channel) and volume_spike and trend_down
        
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals