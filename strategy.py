#!/usr/bin/env python3
"""
12h_1dKAMA_RSI_Trend_Filter
Hypothesis: Use daily KAMA direction for trend filter, RSI for overbought/oversold, and enter on 12h close crossing KAMA. 
Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).
KAMA adapts to volatility, reducing whipsaw in chop. Targets 15-25 trades/year by requiring alignment of daily trend, 
RSI not extreme, and price crossing KAMA on 12h close.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily KAMA (ER=10, fast=2, slow=30)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    vol = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # placeholder, actual calculation below
    # Correct ER calculation
    dir = np.abs(np.diff(close_1d, n=10, prepend=close_1d[:10]))  # direction over 10 periods
    vol = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # incorrect, redo
    # Let's compute ER properly
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[i-9:i+1], prepend=close_1d[i-9])))
    # Avoid loop, use pandas
    close_1d_series = pd.Series(close_1d)
    change = close_1d_series.diff(10).abs()
    volatility = close_1d_series.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = close_1d_series.copy()
    for i in range(1, len(kama)):
        if not np.isnan(sc.iloc[i]):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_1d_series.iloc[i] - kama.iloc[i-1])
        else:
            kama.iloc[i] = kama.iloc[i-1]
    kama_vals = kama.values
    
    # Calculate daily RSI(14)
    delta = close_1d_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_vals = rsi.values
    
    # Align KAMA and RSI to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_vals)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_vals)
    
    # Volume confirmation on 12h: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily KAMA
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # RSI filter: not extreme (avoid overbought/oversold)
        rsi_not_extreme = (rsi_aligned[i] > 30) and (rsi_aligned[i] < 70)
        
        # Entry conditions: price crosses KAMA in direction of trend with volume
        cross_up = close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1]
        cross_down = close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1]
        
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        long_entry = vol_confirm and uptrend and rsi_not_extreme and cross_up
        short_entry = vol_confirm and downtrend and rsi_not_extreme and cross_down
        
        # Exit: opposite cross or trend change
        long_exit = cross_down or (not uptrend)
        short_exit = cross_up or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1dKAMA_RSI_Trend_Filter"
timeframe = "12h"
leverage = 1.0