#!/usr/bin/env python3
"""
Experiment #710: 1d Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 1d Donchian breakouts filtered by weekly pivot levels (from 1w data) and volume confirmation
captures institutional breakout moves with proper higher-timeframe alignment. Weekly pivot direction 
provides regime filter: long only when price > weekly pivot, short only when price < weekly pivot.
Uses discrete position sizing (0.25) to minimize fee churn. Works in both bull/bear markets via 
weekly pivot regime filter. Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_710_1d_donchian20_1w_pivot_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot calculation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot from weekly data (using prior week's OHLC)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    lookback = 1  # 1 weekly bar = prior week
    if len(high_1w) >= lookback:
        # Prior week's OHLC (shifted by 1 to avoid look-ahead)
        week_high = pd.Series(high_1w).shift(1).values
        week_low = pd.Series(low_1w).shift(1).values
        week_close = pd.Series(close_1w).shift(1).values
        
        # Weekly pivot point
        weekly_pivot = (week_high + week_low + week_close) / 3.0
        
        # Weekly support/resistance levels (basic)
        weekly_r1 = 2 * weekly_pivot - week_low
        weekly_s1 = 2 * weekly_pivot - week_high
        
        # Align to 1d timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    else:
        # Not enough data - fallback to close price
        weekly_pivot_aligned = close.copy()
        weekly_r1_aligned = close.copy() * 1.02
        weekly_s1_aligned = close.copy() * 0.98
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(donchian_period, 20) + 5  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
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
            
            # Optional: time-based exit after 10 bars (~10 days on 1d) to avoid overtrading
            if bars_since_entry > 10:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long: price breaks above Donchian high AND price > weekly pivot (bullish regime)
            if price > donchian_high[i] and price > weekly_pivot_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian low AND price < weekly pivot (bearish regime)
            elif price < donchian_low[i] and price < weekly_pivot_aligned[i]:
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