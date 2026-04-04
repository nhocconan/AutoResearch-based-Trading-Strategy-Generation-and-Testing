#!/usr/bin/env python3
"""
Experiment #4759: 6h Donchian(20) Breakout + 12h ADX Trend + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with 12h ADX>25 trend filter and volume confirmation (>1.5x average) capture strong momentum moves while avoiding choppy markets. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag. Works in bull markets (breakouts with trend) and bear markets (breakdowns against trend) by using ADX to filter for trending conditions only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4759_6h_donchian20_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for ADX25 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h Indicators: ADX(14) for trend filter ===
    if len(df_12h) >= 14:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # True Range
        tr1 = high_12h[1:] - low_12h[1:]
        tr2 = np.abs(high_12h[1:] - close_12h[:-1])
        tr3 = np.abs(low_12h[1:] - close_12h[:-1])
        tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3)))
        
        # Directional Movement
        up_move = high_12h[1:] - high_12h[:-1]
        down_move = low_12h[:-1] - low_12h[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values using Wilder's smoothing (alpha = 1/period)
        def wilders_smoothing(values, period):
            alpha = 1.0 / period
            result = np.full_like(values, np.nan)
            for i in range(len(values)):
                if np.isnan(result[i-1]) if i > 0 else True:
                    result[i] = values[i]
                else:
                    result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
            return result
        
        atr_12h = wilders_smoothing(tr_12h, 14)
        plus_di_12h = 100 * wilders_smoothing(plus_dm, 14) / atr_12h
        minus_di_12h = 100 * wilders_smoothing(minus_dm, 14) / atr_12h
        dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
        adx_12h = wilders_smoothing(dx_12h, 14)
    else:
        adx_12h = np.full(len(df_12h), np.nan)
    
    # Align HTF ADX to 6h timeframe
    if len(adx_12h) > 0:
        adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    else:
        adx_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
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
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_12h_aligned[i] > 25.0
        
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with trend alignment
        breakout_long = (price >= high_roll[i]) and trending and vol_confirm
        breakout_short = (price <= low_roll[i]) and trending and vol_confirm
        
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