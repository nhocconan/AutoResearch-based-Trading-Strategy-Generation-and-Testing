#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike captures sustainable moves while avoiding false breakouts. Long when close breaks above R1 with volume > 1.5x MA20 and price > 12h EMA50; Short when close breaks below S1 with volume > 1.5x MA20 and price < 12h EMA50. Uses discrete sizing (±0.25) and ATR-based stoploss (signal→0 when price moves against position by 2*ATR). Designed for 20-50 trades/year with BTC/ETH edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    # Need daily high, low, close - get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Previous day high
    prev_low = df_1d['low'].shift(1).values    # Previous day low
    prev_close = df_1d['close'].shift(1).values # Previous day close
    
    # Align to 4h timeframe (each 1d bar = 6 bars of 4h)
    prev_high_4h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_4h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_4h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    rang = prev_high_4h - prev_low_4h
    r1 = prev_close_4h + rang * 1.1 / 12
    s1 = prev_close_4h - rang * 1.1 / 12
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # ATR for stoploss (2*ATR)
    tr1 = pd.Series(high - low)
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Camarilla (need prev day), EMA50 (50), volume MA (20), ATR (14)
    start_idx = max(6, 50, 20, 14)  # 6 for previous day alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i]) or
            np.isnan(prev_high_4h[i]) or np.isnan(prev_low_4h[i]) or np.isnan(prev_close_4h[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        ema_50_val = ema_50_12h_aligned[i]
        atr_val = atr[i]
        
        # Track entry price for stoploss
        if i == start_idx:
            entry_price = close_val if position != 0 else 0
        
        # Update entry price when position changes
        if position == 0 and signals[i] != 0:
            entry_price = close_val
        
        # Stoploss: close beyond 2*ATR from entry
        stop_long = position == 1 and close_val < entry_price - 2.0 * atr_val
        stop_short = position == -1 and close_val > entry_price + 2.0 * atr_val
        
        # Entry conditions
        long_entry = (close_val > r1[i]) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < s1[i]) and vol_spike and (close_val < ema_50_val)
        
        # Exit conditions
        long_exit = stop_long or (close_val < r1[i])  # Exit if stop hit or price returns below R1
        short_exit = stop_short or (close_val > s1[i])  # Exit if stop hit or price returns above S1
        
        if stop_long or stop_short:
            signals[i] = 0.0
            position = 0
        elif long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0