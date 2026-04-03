#!/usr/bin/env python3
"""
Experiment #084: 1d Donchian(20) breakout + 1w EMA trend + volume confirmation

HYPOTHESIS: Daily Donchian(20) breakouts capture significant price moves, filtered by weekly EMA21 trend direction.
Volume confirmation (>1.5x 20-day average) ensures institutional participation. Discrete sizing (0.30) limits drawdown.
Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out by weekly EMA).
Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_084_1d_donchian_1w_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for EMA21 trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        ema_21 = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    else:
        ema_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (30% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0  # 1 for long, -1 for short
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if np.isnan(ema_21_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter ---
        # Weekly EMA21 trend: price above EMA = uptrend, below = downtrend
        uptrend = close[i] > ema_21_aligned[i]
        downtrend = close[i] < ema_21_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_high[i]
        bearish_breakout = close[i] < donch_low[i]
        
        # Only take breakouts in direction of weekly trend
        if bullish_breakout and uptrend and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif bearish_breakout and downtrend and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals