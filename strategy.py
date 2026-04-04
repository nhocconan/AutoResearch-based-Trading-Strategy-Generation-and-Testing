#!/usr/bin/env python3
"""
Experiment #4694: 1h Donchian(20) Breakout + 4h/1d EMA Trend + Volume Confirmation + Session Filter
HYPOTHESIS: 1h price breaking Donchian(20) channels with volume confirmation (>1.5x avg volume) and aligned with 4h EMA20 and 1d EMA50 trend filters captures momentum while minimizing whipsaws. Session filter (08-20 UTC) reduces noise trades. Uses 4h/1d for signal direction, 1h only for entry timing. Target: 60-150 total trades over 4 years = 15-37/year for 1h. Works in both bull (breakouts with volume) and bear (short breakdowns with volume) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4694_1h_donchian20_4h_1d_ema_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute HTF: 4h and 1d data for EMA trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h Indicators: EMA20 for trend filter ===
    if len(df_4h) >= 20:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        ema_4h = np.full(len(df_4h), np.nan)
    
    # Align HTF EMA20 to 1h timeframe
    if len(ema_4h) > 0:
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: EMA50 for trend filter ===
    if len(df_1d) >= 50:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF EMA50 to 1h timeframe
    if len(ema_1d) > 0:
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian(20) from prior 20 bars ===
    # Use prior 20 bars' high/low (shifted by 1 to avoid look-ahead)
    ph = np.concatenate([[np.nan] * 20, high[:-20]])  # prior 20 bars high
    pl = np.concatenate([[np.nan] * 20, low[:-20]])   # prior 20 bars low
    
    # Rolling max/min of prior 20 bars
    donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 50, 14)  # Donchian, 4h EMA, 1d EMA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
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
        # Volume filter: confirmation for breakouts (>1.5x)
        vol_breakout = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions
        breakout_long = price > donchian_high[i] and vol_breakout
        breakout_short = price < donchian_low[i] and vol_breakout
        
        # Trend filters: 4h EMA20 and 1d EMA50
        trend_filter_long = (price > ema_4h_aligned[i]) and (price > ema_1d_aligned[i])
        trend_filter_short = (price < ema_4h_aligned[i]) and (price < ema_1d_aligned[i])
        
        # Final entry conditions: breakout + volume + trend filters
        if breakout_long and trend_filter_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short and trend_filter_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals