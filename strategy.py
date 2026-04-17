#!/usr/bin/env python3
"""
4h_RSI_MeanReversion_TrendFilter
Hypothesis: On 4h, enter long when RSI(14) < 30 with 12h EMA(50) support and volume confirmation; short when RSI > 70 with resistance and volume. Uses mean reversion in overextended conditions filtered by higher timeframe trend to avoid counter-trend trades in strong moves. Designed for 20-40 trades/year to minimize fee drag and work in both bull/bear regimes via trend alignment.
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
    
    # === 12h data for trend and volume ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h volume average (20-period) for confirmation
    vol_avg20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg20_12h)
    
    signals = np.zeros(n)
    
    # Warmup: covers RSI and 12h indicators
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_avg20_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 12h volume
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = vol_12h_current > 1.5 * vol_avg20_12h_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: RSI oversold + above 12h EMA50 + volume
            if rsi[i] < 30 and close[i] > ema50_12h_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI overbought + below 12h EMA50 + volume
            elif rsi[i] > 70 and close[i] < ema50_12h_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse when RSI returns to neutral zone (40-60)
        elif position == 1:
            if rsi[i] > 40:  # exit long when RSI recovers
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if rsi[i] < 60:  # exit short when RSI declines
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_MeanReversion_TrendFilter"
timeframe = "4h"
leverage = 1.0