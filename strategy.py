#!/usr/bin/env python3
"""
4h_RSI2_Regime_Breakout_v2
Tightened RSI(2) breakout with volume confirmation and regime filter:
- Long when RSI(2) < 10, price breaks above Donchian(20), volume > 1.5x avg, and Choppiness > 61.8 (range)
- Short when RSI(2) > 90, price breaks below Donchian(20), volume > 1.5x avg, and Choppiness > 61.8
- Exit when RSI(2) crosses above 50 (long) or below 50 (short)
- Uses 1d trend filter: only long if price > 200 EMA, only short if price < 200 EMA
- Designed for 20-40 trades/year per symbol with strong edge in ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(2) with proper smoothing
    change = np.diff(close, prepend=close[0])
    gain = np.where(change > 0, change, 0.0)
    loss = np.where(change < 0, -change, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Donchian channels (20-period)
    lookback = 20
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest[i] = np.max(high[i-lookback+1:i+1])
        lowest[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    
    # Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=1).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=1).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=1).min().values
    chop = np.where((highest_high - lowest_low) != 0, 
                    100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14),
                    50)
    
    # 1d trend filter (200 EMA)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_4h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need sufficient data for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i]) or np.isnan(ema200_1d_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x average
        vol_condition = volume[i] > 1.5 * vol_ma[i]
        
        # Regime condition: choppy market (Choppiness > 61.8)
        regime_condition = chop[i] > 61.8
        
        # Trend filter: price above/below 200 EMA
        price_above_ema200 = close[i] > ema200_1d_4h[i]
        price_below_ema200 = close[i] < ema200_1d_4h[i]
        
        if position == 0:
            # Long: RSI(2) < 10, breakout above Donchian high, volume, regime, and trend
            if (rsi[i] < 10 and close[i] > highest[i-1] and vol_condition and 
                regime_condition and price_above_ema200):
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90, breakdown below Donchian low, volume, regime, and trend
            elif (rsi[i] > 90 and close[i] < lowest[i-1] and vol_condition and 
                  regime_condition and price_below_ema200):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI(2) crosses above 50
            if rsi[i] > 50 and rsi[i-1] <= 50:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI(2) crosses below 50
            if rsi[i] < 50 and rsi[i-1] >= 50:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI2_Regime_Breakout_v2"
timeframe = "4h"
leverage = 1.0