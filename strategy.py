#!/usr/bin/env python3
"""
Experiment #4979: 6h Donchian(20) Breakout + 12h Volume Spike + ATR Stoploss
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with 12h volume confirmation (>2x average) capture strong momentum moves while minimizing whipsaw. Uses ATR(14) trailing stop (2.5x) to limit downside. Designed for 12-37 trades/year on 6h timeframe to balance statistical significance with fee drag control. Volume filter from higher timeframe (12h) reduces false breakouts during low-liquidity periods, improving performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4979_6h_donchian20_12h_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h Indicators: Volume MA for confirmation ===
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    else:
        vol_ma_12h = np.full(len(df_12h), np.nan)
    
    # Align HTF 12h volume MA to 6h timeframe
    if len(vol_ma_12h) > 0:
        vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    else:
        vol_ma_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>2.0x from 12h MA)
        vol_ratio = volume[i] / vol_ma_12h_aligned[i] if vol_ma_12h_aligned[i] > 0 else 0
        vol_confirm = vol_ratio > 2.0
        
        # Donchian breakout conditions with volume confirmation
        breakout_long = (price >= high_roll[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and vol_confirm
        
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