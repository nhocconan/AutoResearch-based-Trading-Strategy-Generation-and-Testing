#!/usr/bin/env python3
"""
Experiment #511: 6h ADX(14) trend strength + Williams %R(14) mean reversion + 1d EMA(50) filter
HYPOTHESIS: In ranging markets (ADX < 25), Williams %R extremes (< -80 for long, > -20 for short) capture mean reversion with high win rate. In trending markets (ADX >= 25), we require price to be above/below 1d EMA(50) to align with higher timeframe trend, filtering counter-trend trades. This dual-regime approach works in both bull and bear markets by adapting to market conditions. 6h timeframe reduces fee drag while capturing sufficient moves. Discrete position sizing (0.25) manages drawdown. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_511_6h_adx_williamsr_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA(50) filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Indicators: ADX(14) ===
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    
    # Directional Movement
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
        dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # ADX
    dx = np.zeros(n)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Williams %R(14) and ADX(14) warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx[i]) or np.isnan(williams_r[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Regime Detection: ADX < 25 = ranging, ADX >= 25 = trending ---
        is_ranging = adx[i] < 25
        is_trending = adx[i] >= 25
        
        # --- Mean Reversion Signals (Williams %R extremes) ---
        wr_oversold = williams_r[i] < -80  # Extreme oversold
        wr_overbought = williams_r[i] > -20  # Extreme overbought
        
        # --- Trend Filter (1d EMA(50)) ---
        price_above_ema = price > ema_50_1d_aligned[i]
        price_below_ema = price < ema_50_1d_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss (using 2*ATR) ---
        # Recalculate ATR for exit (could store but recompute for clarity)
        tr_i = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])) if i > 0 else high[i] - low[i]
        atr_i = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values[i]
        
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_i
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_i
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~3 days on 6h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if is_ranging:
            # In ranging markets: mean reversion from Williams %R extremes
            if wr_oversold:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif wr_overbought:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:  # is_trending
            # In trending markets: only trade in direction of 1d EMA(50)
            if wr_oversold and price_above_ema:
                # Oversold but above EMA = long opportunity in uptrend
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif wr_overbought and price_below_ema:
                # Overbought but below EMA = short opportunity in downtrend
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals