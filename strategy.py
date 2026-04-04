#!/usr/bin/env python3
"""
Experiment #4659: 6h Donchian(20) Breakout + 12h Volume Spike + 1d ADX Regime Filter
HYPOTHESIS: 6h price breaking Donchian(20) channels (from prior 20 6h bars) with volume confirmation (>2x 20-period MA) captures momentum. 
Only trade when 12h ADX > 25 (trending regime) to avoid whipsaws in ranging markets. 
Target: 12-37 trades/year on 6h timeframe. Works in bull (breakouts) and bear (trending regime filters out false signals).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4659_6h_donchian20_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h Indicators: ADX(14) for regime filter ===
    if len(df_12h) >= 14:
        # True Range
        tr1 = df_12h['high'].values[1:] - df_12h['low'].values[1:]
        tr2 = np.abs(df_12h['high'].values[1:] - df_12h['close'].values[:-1])
        tr3 = np.abs(df_12h['low'].values[1:] - df_12h['close'].values[:-1])
        tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))))
        
        # Directional Movement
        up_move = df_12h['high'].values[1:] - df_12h['high'].values[:-1]
        down_move = df_12h['low'].values[:-1] - df_12h['low'].values[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR, +DM, -DM
        tr_smooth = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_12h = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    else:
        adx_12h = np.full(len(df_12h), np.nan)
    
    # Align 12h ADX to 6h timeframe
    if len(adx_12h) > 0:
        adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) from prior 20 6h bars ===
    if n >= 20:
        # Use prior 20 periods' high/low (shifted by 1)
        ph = np.concatenate([[np.nan] * 20, high[:-20]])  # prior 20 periods high
        pl = np.concatenate([[np.nan] * 20, low[:-20]])   # prior 20 periods low
        
        # Rolling max/min of prior 20 periods
        donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))))
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
    
    warmup = max(20, 14)  # Donchian, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or np.isnan(adx_aligned[i])):
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
        # Volume filter: confirmation for breakouts (>2.0x)
        vol_breakout = vol_ratio[i] > 2.0
        
        # Regime filter: only trade in trending markets (ADX > 25)
        trending_regime = adx_aligned[i] > 25.0
        
        # Breakout conditions: price breaks Donchian high/low with volume confirmation and trending regime
        breakout_long = price > donchian_high[i] and vol_breakout and trending_regime
        breakout_short = price < donchian_low[i] and vol_breakout and trending_regime
        
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