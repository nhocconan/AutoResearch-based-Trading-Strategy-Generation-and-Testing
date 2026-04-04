#!/usr/bin/env python3
"""
Experiment #3314: 1h Donchian Breakout + 4h/1d Trend Filter + Session Filter
HYPOTHESIS: 1h Donchian(20) breakouts with 4h/1d EMA trend alignment and volume confirmation capture medium-term swings.
Session filter (08-20 UTC) reduces noise. Position size 0.20. Target: 60-150 total trades over 4 years (15-37/year).
Uses 4h/1d for signal direction, 1h only for entry timing to avoid overtrading. Works in bull/bear via trend filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3314_1h_donchian20_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for EMA trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Session filter: 08-20 UTC (pre-compute hours array) ===
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback, 20, 21, 50)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic (ATR-based trailing stop + mean reversion) ---
        if in_position:
            # Simple ATR trailing stop (2.0x ATR)
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                if price < highest_since_entry - 2.0 * (high[i] - low[i]):  # proxy ATR
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif price <= highest_high[i]:  # re-enter Donchian = exit
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                if price > lowest_since_entry + 2.0 * (high[i] - low[i]):  # proxy ATR
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif price >= lowest_low[i]:  # re-enter Donchian = exit
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume spike confirmation (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Trend filters: 4h EMA and 1d EMA must agree
            bullish = price > ema_4h_aligned[i] and price > ema_1d_aligned[i]
            bearish = price < ema_4h_aligned[i] and price < ema_1d_aligned[i]
            
            # Long entry: Donchian breakout + bullish trend + volume
            if price > highest_high[i] and bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Donchian breakdown + bearish trend + volume
            elif price < lowest_low[i] and bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals