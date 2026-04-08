#!/usr/bin/env python3
"""
4h_rsi_pullback_volume_v1
Hypothesis: RSI pullback strategy on 4h timeframe with volume confirmation and EMA trend filter.
Works in both bull and bear markets by taking pullbacks in the direction of the higher timeframe trend.
- Long when RSI < 30 (oversold), price > EMA20 (uptrend), and volume > 1.5x average
- Short when RSI > 70 (overbought), price < EMA20 (downtrend), and volume > 1.5x average
- Uses daily EMA50 as higher timeframe trend filter to avoid counter-trend trades
- Targets 20-40 trades/year to minimize fee drag while capturing high-probability setups
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_pullback_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA20 for 4h trend
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Need enough data for EMA20 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20[i]) or np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or price < EMA20
            if rsi[i] > 70 or close[i] < ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or price > EMA20
            if rsi[i] < 30 or close[i] > ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI < 30 (oversold), price > EMA20 (uptrend), price > daily EMA50 (bullish bias), volume surge
            if (rsi[i] < 30 and 
                close[i] > ema20[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 70 (overbought), price < EMA20 (downtrend), price < daily EMA50 (bearish bias), volume surge
            elif (rsi[i] > 70 and 
                  close[i] < ema20[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals