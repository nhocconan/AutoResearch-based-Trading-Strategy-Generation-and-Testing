#!/usr/bin/env python3
"""
Experiment #095: 6h Elder Ray + 1d Weekly Pivot Trend Filter
HYPOTHESIS: 6h Elder Ray (Bull/Bear Power) combined with 1d weekly pivot direction creates a robust trend-following system that works in both bull and bear markets. Weekly pivot provides structural support/resistance from higher timeframe, while Elder Ray measures underlying bull/bear strength on the 6h chart. Volume confirmation filters weak breakouts. Target: 75-150 total trades over 4 years (19-37/year). Discrete sizing: 0.25. ATR stoploss (2.5x) controls drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_095_6h_elder_ray_1d_weekly_pivot_v1"
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
    
    # Calculate weekly pivot points from prior week's OHLC
    def calculate_weekly_pivot(high_arr, low_arr, close_arr):
        # Need prior week's data - use rolling window of 5 days (1 week)
        if len(high_arr) < 5:
            return np.full_like(high_arr, np.nan), np.full_like(high_arr, np.nan), np.full_like(high_arr, np.nan)
        
        # Weekly high/low/close (prior completed week)
        weekly_high = pd.Series(high_arr).rolling(window=5, min_periods=5).max().shift(1).values
        weekly_low = pd.Series(low_arr).rolling(window=5, min_periods=5).min().shift(1).values
        weekly_close = pd.Series(close_arr).rolling(window=5, min_periods=5).last().shift(1).values
        
        # Weekly pivot: (H + L + C) / 3
        pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Support/resistance levels
        r1 = 2 * pivot - weekly_low
        s1 = 2 * pivot - weekly_high
        r2 = pivot + (weekly_high - weekly_low)
        s2 = pivot - (weekly_high - weekly_low)
        
        return pivot, r1, r2, s1, s2
    
    pivot_1d, r1_1d, r2_1d, s1_1d, s2_1d = calculate_weekly_pivot(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Align pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # === 6h Indicators: Elder Ray (Bull Power / Bear Power) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for EMA stability and pivot calculation
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Weekly Pivot Trend Direction ---
        # Above weekly pivot and R1 = bullish bias
        # Below weekly pivot and S1 = bearish bias
        pivot_bullish = price > pivot_1d_aligned[i] and price > r1_1d_aligned[i]
        pivot_bearish = price < pivot_1d_aligned[i] and price < s1_1d_aligned[i]
        
        # --- Elder Ray Strength ---
        # Bull Power > 0 indicates bulls in control
        # Bear Power < 0 indicates bears in control
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] < 0
        
        # --- Volume Confirmation ---
        volume_spike = vol_ratio[i] > 1.5
        
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
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long: Bullish pivot bias + Bull Power positive + volume confirmation
        if pivot_bullish and bull_strong and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Bearish pivot bias + Bear Power negative + volume confirmation
        elif pivot_bearish and bear_strong and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals