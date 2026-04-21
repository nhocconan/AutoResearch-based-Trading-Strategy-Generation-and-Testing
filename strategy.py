#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
Hypothesis: Camarilla pivot R1/S1 breakout on 1d with volume confirmation (>1.5x 20-bar MA) and ATR-based stoploss (2.5x ATR) works on daily timeframe for BTC and ETH in both bull and bear markets. Uses 1w timeframe for trend filter (price > weekly EMA34) to avoid counter-trend trades. Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data once for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla pivot levels from previous day (1d)
    # We need to get the previous day's OHLC for each bar
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_prices = prices['open'].values
    
    # Shift to get previous day's values (since we're on 1d timeframe)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan
    
    # Camarilla pivot calculations
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    R1 = pivot + (range_hl * 1.1 / 12)
    S1 = pivot - (range_hl * 1.1 / 12)
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>1.5x average)
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Trend filter: only long in uptrend (price > weekly EMA34), only short in downtrend
        uptrend = price > ema34_1w_aligned[i]
        downtrend = price < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and uptrend
            if price > R1[i] and volume_ok and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume and downtrend
            elif price < S1[i] and volume_ok and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit: price closes below S1 (reversal) or stoploss hit
            if price < S1[i] or price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R1 (reversal) or stoploss hit
            if price > R1[i] or price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "1d"
leverage = 1.0