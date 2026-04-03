#!/usr/bin/env python3
"""
Experiment #621: 4h Donchian(20) breakout + 1d EMA(50) trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts aligned with 1d EMA(50) trend capture medium-term momentum with reduced whipsaw. Volume confirmation ensures institutional participation. Target: 75-200 total trades over 4 years via tight entry conditions (Donchian breakout + HTF trend + volume spike). This version tightens volume confirmation to 2.0x (from 1.8x) and adds ATR-based profit taking to reduce trade frequency and improve Sharpe in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_621_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA(50) trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    max_favorable_price = 0.0  # For trailing profit taking
    
    warmup = 50  # sufficient for Donchian and EMA calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- HTF Trend Filter: 1d EMA(50) direction ---
        # Long only when price above 1d EMA(50) (uptrend)
        # Short only when price below 1d EMA(50) (downtrend)
        ema_trend_up = price > ema_1d_aligned[i]
        ema_trend_down = price < ema_1d_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss and profit taking ---
        if in_position:
            bars_since_entry += 1
            
            # Update max favorable price for trailing
            if position_side > 0:  # Long position
                max_favorable_price = max(max_favorable_price, price)
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                # Profit take: reduce to half position at 2R profit
                profit_level = entry_price + 2.0 * atr[i] * 2.0  # 2R = 4*ATR
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    max_favorable_price = 0.0
                    signals[i] = 0.0
                    continue
                elif high[i] > profit_level and bars_since_entry > 2:
                    # Take half profit
                    signals[i] = position_side * SIZE * 0.5
                    continue
            else:  # Short position
                max_favorable_price = min(max_favorable_price, price)
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                # Profit take: reduce to half position at 2R profit
                profit_level = entry_price - 2.0 * atr[i] * 2.0  # 2R = 4*ATR
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    max_favorable_price = 0.0
                    signals[i] = 0.0
                    continue
                elif low[i] < profit_level and bars_since_entry > 2:
                    # Take half profit
                    signals[i] = position_side * SIZE * 0.5
                    continue
            
            # Optional: time-based exit after 8 bars (~32h on 4h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                max_favorable_price = 0.0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up + 1d EMA(50) uptrend
            if breakout_up and ema_trend_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                max_favorable_price = price
                signals[i] = SIZE
            # Short: Donchian breakout down + 1d EMA(50) downtrend
            elif breakout_down and ema_trend_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                max_favorable_price = price
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals