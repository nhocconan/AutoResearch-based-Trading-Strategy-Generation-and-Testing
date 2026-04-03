#!/usr/bin/env python3
"""
Experiment #134: 1h Volume Spike + 4h Donchian Breakout + 1d Trend Filter

HYPOTHESIS: On 1h timeframe, combine 4h Donchian breakout for trend direction with 1h volume spike confirmation and 1d EMA200 filter to capture high-probability momentum moves. The 4h Donchian(20) provides structure and filters noise, 1h volume spike (>2.0x average) confirms institutional participation, and 1d EMA200 ensures alignment with higher timeframe trend. Targets 60-150 trades over 4 years (15-37/year) by requiring confluence of three filters, minimizing fee drag while capturing strong trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_vol_ema200_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for Donchian channel (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 20:
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        # Donchian(20): highest high and lowest low of last 20 bars
        highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
        lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
        # Align to 1h timeframe (shifted by 1 HTF bar for completed bars only)
        highest_20_aligned = align_htf_to_ltf(prices, df_4h, highest_20)
        lowest_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_20)
    else:
        highest_20_aligned = np.full(n, np.nan)
        lowest_20_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for EMA200 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    else:
        ema_200_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators ===
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = np.zeros(n, dtype=bool)
    volume_spike[20:] = volume[20:] > (vol_ma_20[20:] * 2.0)
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Breakout conditions
        bullish_breakout = close[i] > highest_20_aligned[i]
        bearish_breakout = close[i] < lowest_20_aligned[i]
        
        # Trend filter: price above/below 1d EMA200
        price_above_ema = close[i] > ema_200_1d_aligned[i]
        price_below_ema = close[i] < ema_200_1d_aligned[i]
        
        # Entry conditions
        long_entry = bullish_breakout and price_above_ema and volume_spike[i]
        short_entry = bearish_breakout and price_below_ema and volume_spike[i]
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>