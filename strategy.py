#!/usr/bin/env python3
"""
Experiment #4727: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (from 1d HTF) with volume confirmation (>1.5x average) capture strong momentum moves while minimizing false breakouts. Uses ATR(14) trailing stop (2.0x) for risk control. Designed for 12-37 trades/year on 6h timeframe to balance statistical significance with fee drag minimization. Weekly pivot provides structural bias: price above weekly pivot = bullish bias (long breakouts only), below = bearish bias (short breakouts only). This adaptive bias helps in both bull and bear markets by aligning with higher timeframe structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4727_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Pivot (using prior week's OHLC) ===
    if len(df_1d) >= 5:
        # Calculate weekly OHLC from daily data
        # We'll compute weekly pivot using the prior completed week's data
        weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)  # Prior week's high
        weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)    # Prior week's low
        weekly_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)  # Prior week's close
        
        # Weekly Pivot Point = (Prior Week High + Prior Week Low + Prior Week Close) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Convert to numpy arrays
        weekly_pivot_vals = weekly_pivot.values
        weekly_high_vals = weekly_high.values
        weekly_low_vals = weekly_low.values
    else:
        weekly_pivot_vals = np.full(len(df_1d), np.nan)
        weekly_high_vals = np.full(len(df_1d), np.nan)
        weekly_low_vals = np.full(len(df_1d), np.nan)
    
    # Align HTF weekly pivot to 6h timeframe
    if len(weekly_pivot_vals) > 0:
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_vals)
        weekly_high_aligned = align_htf_to_ltf(prices, df_1d, weekly_high_vals)
        weekly_low_aligned = align_htf_to_ltf(prices, df_1d, weekly_low_vals)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
        weekly_high_aligned = np.full(n, np.nan)
        weekly_low_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for trailing stoploss ===
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
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Determine bias from weekly pivot: price above pivot = bullish, below = bearish
        bullish_bias = price > weekly_pivot_aligned[i]
        bearish_bias = price < weekly_pivot_aligned[i]
        
        # Donchian breakout conditions with weekly pivot bias
        breakout_long = (price >= high_roll[i]) and bullish_bias and vol_confirm
        breakout_short = (price <= low_roll[i]) and bearish_bias and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals