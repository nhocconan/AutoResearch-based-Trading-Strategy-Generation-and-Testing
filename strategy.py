#!/usr/bin/env python3
"""
Experiment #4654: 1h Donchian(20) Breakout + 4h Volume Spike + Session Filter
HYPOTHESIS: 1h price breaking Donchian(20) channels (from prior 20 1h bars) with volume confirmation (>2x 20-period average) and during active session (08-20 UTC) captures momentum. 
HTF 4h Donchian(20) provides directional bias: only take longs when price > 4h Donchian mid, shorts when price < 4h Donchian mid. 
Reduces whipsaw in ranging markets. Target: 60-150 total trades over 4 years = 15-37/year on 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4654_1h_donchian20_4h_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # Precompute HTF: 4h data for directional bias
    df_4h = get_htf_data(prices, '4h')
    
    # === 4h Indicators: Donchian(20) from prior 20 periods ===
    if len(df_4h) >= 20:
        # Use prior 20 periods' high/low (shifted by 1)
        ph = np.concatenate([[np.nan] * 20, df_4h['high'].values[:-20]])  # prior 20 periods high
        pl = np.concatenate([[np.nan] * 20, df_4h['low'].values[:-20]])   # prior 20 periods low
        
        # Rolling max/min of prior 20 periods
        donchian_high_4h = pd.Series(ph).rolling(window=20, min_periods=20).max().values
        donchian_low_4h = pd.Series(pl).rolling(window=20, min_periods=20).min().values
        donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2.0
    else:
        donchian_high_4h = np.full(len(df_4h), np.nan)
        donchian_low_4h = np.full(len(df_4h), np.nan)
        donchian_mid_4h = np.full(len(df_4h), np.nan)
    
    # Align HTF indicators to 1h timeframe
    if len(donchian_high_4h) > 0:
        dh_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
        dl_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
        dm_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    else:
        dh_4h_aligned = np.full(n, np.nan)
        dl_4h_aligned = np.full(n, np.nan)
        dm_4h_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian(20) breakout ===
    lookback = 20
    if n >= lookback:
        # Use prior 20 periods' high/low (shifted by 1)
        ph_1h = np.concatenate([[np.nan] * lookback, high[:-lookback]])  # prior 20 periods high
        pl_1h = np.concatenate([[np.nan] * lookback, low[:-lookback]])   # prior 20 periods low
        
        # Rolling max/min of prior 20 periods
        donchian_high_1h = pd.Series(ph_1h).rolling(window=lookback, min_periods=lookback).max().values
        donchian_low_1h = pd.Series(pl_1h).rolling(window=lookback, min_periods=lookback).min().values
    else:
        donchian_high_1h = np.full(n, np.nan)
        donchian_low_1h = np.full(n, np.nan)
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high_1h[i]) or np.isnan(donchian_low_1h[i]) or 
            np.isnan(dh_4h_aligned[i]) or np.isnan(dl_4h_aligned[i]) or 
            np.isnan(dm_4h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: spike for breakout confirmation (>2.0x)
        vol_spike = vol_ratio[i] > 2.0
        
        # Breakout conditions: price breaks 1h Donchian high/low with volume spike
        breakout_long = price > donchian_high_1h[i] and vol_spike
        breakout_short = price < donchian_low_1h[i] and vol_spike
        
        # HTF Directional Bias: only trade in direction of 4h trend
        bias_long = price > dm_4h_aligned[i]   # Above 4h Donchian mid = bullish bias
        bias_short = price < dm_4h_aligned[i]  # Below 4h Donchian mid = bearish bias
        
        # Entry: breakout in direction of HTF bias
        if breakout_long and bias_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short and bias_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
</trading_assistant>