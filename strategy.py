#!/usr/bin/env python3
"""
Experiment #4714: 1h Donchian Breakout + 4h/1d Trend + Volume + Session Filter
HYPOTHESIS: On 1h timeframe, price breaking Donchian(20) channels with 4h EMA50 trend alignment and 1d EMA200 filter captures institutional breakouts. Volume confirmation (>1.5x) ensures participation. Session filter (08-20 UTC) reduces noise. Target 60-150 trades over 4 years by using tight 4h/1d trend filters and volume spike requirement. Works in bull markets (breakouts with trend) and bear markets (mean reversion at channel extremes during low volatility).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4714_1h_donchian20_4h_1d_trend_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h Indicators: EMA50 for trend filter ===
    if len(df_4h) >= 50:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False).mean().values
    else:
        ema_4h = np.full(len(df_4h), np.nan)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d Indicators: EMA200 for higher timeframe trend ===
    if len(df_1d) >= 200:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    else:
        ema_1d = np.full(len(df_1d), np.nan)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 1h Indicators: Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Session filter: 08-20 UTC (pre-compute hours) ===
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback, 20, 50, 200)  # Donchian, Vol MA, 4h EMA, 1d EMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        hour = hours[i]
        
        # Session filter: only trade 08-20 UTC
        if hour < 8 or hour > 20:
            if in_position:
                # Exit at session close if still in position
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
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
        # Volume filter: confirmation (>1.5x average)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions
        breakout_long = (price >= highest[i]) and vol_confirm
        breakout_short = (price <= lowest[i]) and vol_confirm
        
        # Trend filters: 4h EMA50 and 1d EMA200 alignment
        uptrend_4h = price > ema_4h_aligned[i]
        uptrend_1d = price > ema_1d_aligned[i]
        downtrend_4h = price < ema_4h_aligned[i]
        downtrend_1d = price < ema_1d_aligned[i]
        
        # Final entry conditions: breakout with trend alignment
        if breakout_long and uptrend_4h and uptrend_1d:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short and downtrend_4h and downtrend_1d:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
</think>