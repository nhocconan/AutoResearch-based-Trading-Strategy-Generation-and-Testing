#!/usr/bin/env python3
"""
Experiment #4914: 1h Donchian(20) Breakout + 4h/1d EMA Trend + Volume Spike + Session Filter
HYPOTHESIS: On 1h timeframe, Donchian(20) breakouts in direction of 4h EMA50 and 1d EMA200 trend with volume confirmation (>1.5x average) capture momentum moves. Session filter (08-20 UTC) reduces noise. Position size fixed at 0.20 to limit drawdown. Designed for 15-37 trades/year on 1h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakdowns against trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4914_1h_donchian20_4h_1d_ema_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    # Precompute HTF: 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h Indicators: EMA50 for trend filter ===
    if len(df_4h) >= 50:
        close_4h = df_4h['close'].values
        ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_4h = np.full(len(df_4h), np.nan)
    
    # Align HTF EMA50 to 1h timeframe
    if len(ema_4h) > 0:
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: EMA200 for trend filter ===
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    else:
        ema_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF EMA200 to 1h timeframe
    if len(ema_1d) > 0:
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Session filter: 08-20 UTC ===
    # prices.index is already DatetimeIndex with hour attribute
    hours = prices.index.hour
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
    
    warmup = max(20, 20, 50, 200)  # Donchian, Volume MA, 4h EMA50, 1d EMA200 warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: reverse signal or stoploss ---
        if in_position:
            # Reverse signal: exit if opposite breakout with volume
            reverse_long = (position_side == -1 and price >= high_roll[i] and vol_ratio[i] > 1.5)
            reverse_short = (position_side == 1 and price <= low_roll[i] and vol_ratio[i] > 1.5)
            
            if reverse_long or reverse_short:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = SIZE if position_side == 1 else -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with trend alignment (both 4h and 1d EMA)
        # Long: price breaks above Donchian high AND above both EMAs
        breakout_long = (price >= high_roll[i]) and (price > ema_4h_aligned[i]) and (price > ema_1d_aligned[i]) and vol_confirm
        # Short: price breaks below Donchian low AND below both EMAs
        breakout_short = (price <= low_roll[i]) and (price < ema_4h_aligned[i]) and (price < ema_1d_aligned[i]) and vol_confirm
        
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