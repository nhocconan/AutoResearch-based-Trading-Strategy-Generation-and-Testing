#!/usr/bin/env python3
"""
12h_Trend_Pullback_EMA_RSI
Hypothesis: On 12h timeframe, buy pullbacks to EMA20 in uptrend (price>EMA50) and sell rallies to EMA20 in downtrend (price<EMA50), using RSI(14) for overbought/oversold confirmation and volume filter to avoid chop. Works in both bull (buy dips) and bear (sell rallies) markets. Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA20 and EMA50 for trend and pullback levels
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14) on 1d close
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Warmup for EMA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema20 = ema20_1d_aligned[i]
        ema50 = ema50_1d_aligned[i]
        rsi = rsi_1d_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: pullback to EMA20 in uptrend, not overbought
            if price > ema50 and abs(price - ema20) < ema20 * 0.005 and rsi < 70 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: rally to EMA20 in downtrend, not oversold
            elif price < ema50 and abs(price - ema20) < ema20 * 0.005 and rsi > 30 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: trend change or overextended
            if price < ema50 or rsi > 80:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: trend change or overextended
            if price > ema50 or rsi < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Trend_Pullback_EMA_RSI"
timeframe = "12h"
leverage = 1.0