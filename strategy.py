#!/usr/bin/env python3
"""
Experiment #4916: 12h Donchian(20) Breakout + 1d Volume Spike + Chop Regime Filter
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts with volume confirmation (>2x average) and 
choppiness regime filter (CHOP > 61.8 = ranging) capture sustained moves while avoiding whipsaws. 
Uses ATR(14) trailing stop (2.5x) for risk management. Designed for 12-37 trades/year to minimize 
fee drag while maintaining statistical significance. Works in bull markets (breakouts with momentum) 
and bear markets (breakdowns with volume) by requiring both price expansion and participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4916_12h_donchian20_1d_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Choppiness Index (CHOP) ===
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1d = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Highest high and lowest low over 14 periods
        hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
        sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
        chop_raw = 100 * np.log10(sum_atr_14 / np.maximum(hh_14 - ll_14, 1e-10)) / np.log10(14)
        chop_1d = chop_raw  # Already in 0-100 range
    else:
        chop_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF CHOP to 12h timeframe
    if len(chop_1d) > 0:
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        chop_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume confirmation (2x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 14, 14)  # Donchian, Volume MA, ATR, CHOP warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        regime_filter = chop_1d_aligned[i] > 61.8
        
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions
        breakout_long = (price >= high_roll[i]) and vol_confirm and regime_filter
        breakout_short = (price <= low_roll[i]) and vol_confirm and regime_filter
        
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