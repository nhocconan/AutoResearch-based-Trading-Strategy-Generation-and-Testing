#!/usr/bin/env python3
"""
Experiment #3994: 1h Donchian(20) breakout + 4h/1d trend alignment + volume confirmation
HYPOTHESIS: 1h Donchian breakouts aligned with 4h/1d EMA200 trend capture sustained moves with less whipsaw.
Volume > 1.5x MA(20) confirms breakout strength. Uses 4h/1d for signal direction, 1h only for entry timing.
Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years (15-37/year).
Works in bull/bear via multi-timeframe EMA200 alignment (avoids counter-trend trades).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3994_1h_donchian20_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for EMA200 trend ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 200:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for EMA200 trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 200:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback_dc + 1, 20, 200 + 10)  # DC lookback, vol MA, EMA buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse on opposite signal ---
        if in_position:
            # Determine current signal direction
            bullish_bias = price > ema_4h_aligned[i] and price > ema_1d_aligned[i]
            bearish_bias = price < ema_4h_aligned[i] and price < ema_1d_aligned[i]
            
            # Exit long if bearish alignment
            if position_side > 0 and not bullish_bias:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            # Exit short if bullish alignment
            elif position_side < 0 and not bearish_bias:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = SIZE if position_side > 0 else -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine trend alignment from 4h/1d EMA200
            bullish_bias = price > ema_4h_aligned[i] and price > ema_1d_aligned[i]
            bearish_bias = price < ema_4h_aligned[i] and price < ema_1d_aligned[i]
            
            # Breakout conditions
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            if bullish_bias and breakout_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif bearish_bias and breakout_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals