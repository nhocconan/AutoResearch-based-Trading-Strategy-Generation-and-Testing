#!/usr/bin/env python3
"""
Experiment #767: 6h Donchian Breakout + Weekly Pivot Direction + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 6h provide clean trend entries, filtered by weekly pivot direction (price above/below weekly pivot = bull/bear bias) and volume confirmation (>2x average). This combines structure (Donchian), regime (weekly pivot), and momentum (volume spike) to work in both bull and bear markets. Weekly pivot acts as a dynamic support/resistance filter - only take long breaks above weekly pivot in bull bias, short breaks below in bear bias. Uses discrete position sizing (0.25) with ATR stoploss.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_767_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from daily OHLC (using last 5 days = 1 week)
    def calculate_weekly_pivot(high, low, close):
        # Need at least 5 days for weekly
        if len(high) < 5:
            return np.full(len(high), np.nan), np.full(len(high), np.nan), np.full(len(high), np.nan)
        
        # Rolling window of 5 days (1 week)
        weekly_high = pd.Series(high).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(low).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(close).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot point: (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly R1: 2*P - L
        weekly_r1 = 2 * weekly_pivot - weekly_low
        # Weekly S1: 2*P - H
        weekly_s1 = 2 * weekly_pivot - weekly_high
        
        return weekly_pivot, weekly_r1, weekly_s1
    
    weekly_pivot_1d, weekly_r1_1d, weekly_s1_1d = calculate_weekly_pivot(high_1d, low_1d, close_1d)
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_1d)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1_1d)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1_1d)
    
    # === 6h Indicators: Donchian Channel(20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
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
    
    warmup = max(20, 20)  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
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
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long: Price breaks above Donchian upper AND price above weekly pivot (bullish bias)
            if close[i] > donch_upper[i] and price > weekly_pivot_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Price breaks below Donchian lower AND price below weekly pivot (bearish bias)
            elif close[i] < donch_lower[i] and price < weekly_pivot_aligned[i]:
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