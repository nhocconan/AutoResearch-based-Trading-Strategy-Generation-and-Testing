#!/usr/bin/env python3
"""
1h_4h1dTrend_Momentum_Entry
Hypothesis: Use 4h trend (EMA21>EMA50) and 1d trend (price>EMA50) as directional filters, enter on 1h momentum bursts when RSI crosses above 60 (long) or below 40 (short) with volume > 1.5x 20-period average. Exit on opposite RSI cross. Designed for low trade frequency (~15-35/year) by requiring multi-timeframe alignment and volume confirmation, reducing noise in choppy markets while capturing momentum in trending regimes.
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA21 and EMA50 for trend
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMAs to 1h timeframe
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h trend: bullish when EMA21 > EMA50
    trend_4h_bull = ema21_4h_aligned > ema50_4h_aligned
    trend_4h_bear = ema21_4h_aligned < ema50_4h_aligned
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d trend: bullish when price > EMA50
    trend_1d_bull = close > ema50_1d_aligned
    trend_1d_bear = close < ema50_1d_aligned
    
    # 1h RSI(14) for momentum entry
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: multi-timeframe trend alignment + momentum + volume
        long_entry = (trend_4h_bull[i] and trend_1d_bull[i] and 
                      rsi[i] > 60 and rsi[i-1] <= 60 and volume_surge[i])
        short_entry = (trend_4h_bear[i] and trend_1d_bear[i] and 
                       rsi[i] < 40 and rsi[i-1] >= 40 and volume_surge[i])
        
        # Exit on opposite RSI cross
        long_exit = rsi[i] < 40 and rsi[i-1] >= 40
        short_exit = rsi[i] > 60 and rsi[i-1] <= 60
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.20  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.20   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h1dTrend_Momentum_Entry"
timeframe = "1h"
leverage = 1.0