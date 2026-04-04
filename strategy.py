#!/usr/bin/env python3
"""
Experiment #4614: 1h Donchian Breakout + 4h EMA Trend + Volume Confirmation
HYPOTHESIS: 1h price breaking 20-period Donchian channels with 4h EMA(21) trend alignment and volume (>1.5x average) captures momentum in both bull and bear markets. Uses 4h/1d HTF for signal direction (trend filter and structure), 1h only for entry timing precision. Session filter (08-20 UTC) reduces noise. Discrete sizing (0.20) and ATR trailing stop (2.0x) manage risk. Target: 60-150 total trades over 4 years = 15-37/year on 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4614_1h_donchian20_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 4h EMA(21) for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 21:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)  # auto shift(1)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # Precompute HTF: 1d Donchian(20) for structure (using prior day to avoid look-ahead)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        # Use prior day's high/low for Donchian (shifted by 1)
        prev_high = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
        prev_low = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
        # Calculate rolling max/min on prior day data
        high_series = pd.Series(prev_high)
        low_series = pd.Series(prev_low)
        donchian_high = high_series.rolling(window=20, min_periods=20).max().values
        donchian_low = low_series.rolling(window=20, min_periods=20).min().values
        donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
        donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    else:
        donchian_high_aligned = np.full(n, np.nan)
        donchian_low_aligned = np.full(n, np.nan)
    
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
    
    # Session filter: 08-20 UTC (pre-compute hours for efficiency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 14, 21)  # Volume MA, ATR, EMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
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
        # Volume confirmation: >1.5x average volume
        vol_confirm = vol_ratio[i] > 1.5
        
        # Breakout conditions: price breaks Donchian levels
        breakout_long = price > donchian_high_aligned[i] and vol_confirm
        breakout_short = price < donchian_low_aligned[i] and vol_confirm
        
        # Trend filter: 4h EMA(21) direction
        # For long: price above 4h EMA (uptrend)
        # For short: price below 4h EMA (downtrend)
        trend_long = price > ema_4h_aligned[i]
        trend_short = price < ema_4h_aligned[i]
        
        # Combine: breakout in direction of 4h trend
        if breakout_long and trend_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short and trend_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals