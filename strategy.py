#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_volume_atr_v1
Strategy: 4h Donchian breakout with volume confirmation and ATR stop
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses 4h Donchian channel breakout (20-period) with volume confirmation (>1.5x average) and ATR-based stop. Filtered by 1d EMA50 trend. Designed to capture strong trending moves with controlled risk. Low trade frequency (~20-40/year) avoids fee drag. Works in both bull (breakouts up) and bear (breakouts down) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # ATR for stop loss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i]) or np.isnan(atr[i])):
            # Maintain current position if valid, else flat
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price_close > ema_50_1d_aligned[i]
        downtrend = price_close < ema_50_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = price_close > donchian_high[i]
        breakout_down = price_close < donchian_low[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Stop loss conditions (ATR-based)
        stop_long = position == 1 and price_close < entry_price - 2.0 * atr[i]
        stop_short = position == -1 and price_close > entry_price + 2.0 * atr[i]
        
        # Trading logic
        if breakout_up and vol_confirmed and uptrend and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif breakout_down and vol_confirmed and downtrend and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif stop_long or stop_short:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals