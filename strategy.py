#!/usr/bin/env python3
"""
4h Choppiness Index + Momentum Filter
Hypothesis: In choppy markets (CHOP > 61.8), mean reversion works using momentum extremes.
Long when CHOP > 61.8 and RSI < 30, short when CHOP > 61.8 and RSI > 70.
Use 1d EMA200 as trend filter: only long when price > EMA200 in chop, short when price < EMA200.
Exit when RSI reverts to 50 or stoploss hits. Works in ranging markets (2025+).
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "choppiness_momentum_4h_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 200-period EMA for daily trend filter
    ema_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14)
    chop_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Avoid division by zero
    range_chop = highest_high - lowest_low
    range_chop = np.where(range_chop == 0, 1e-10, range_chop)
    
    chop = 100 * np.log10(atr_sum / range_chop) / np.log10(chop_period)
    
    # RSI (14)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.where(avg_loss == 0, 100, avg_gain / (avg_loss + 1e-10))
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.7 * vol_ma)  # Require at least 70% of average volume
    
    # ATR for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(chop_period, rsi_period) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI reverts to 50 OR stoploss
            if rsi[i] >= 50 or close[i] <= entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI reverts to 50 OR stoploss
            if rsi[i] <= 50 or close[i] >= entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Choppy market + RSI extreme + trend filter
            choppy = chop[i] > 61.8
            
            long_setup = choppy and (rsi[i] < 30) and (close[i] > ema_1d_aligned[i]) and vol_filter[i]
            short_setup = choppy and (rsi[i] > 70) and (close[i] < ema_1d_aligned[i]) and vol_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals