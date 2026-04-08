#!/usr/bin/env python3
"""
6h 4-Hour Momentum with Weekly Trend Filter and Volume Confirmation
Hypothesis: 4-hour momentum captures short-term swings while weekly EMA ensures alignment with long-term trend, reducing whipsaws in both bull and bear markets. Volume confirms conviction. Target: 15-30 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_4h_momentum_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for momentum
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 4h RSI(14) for momentum
    delta = pd.Series(close_4h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_values = rsi_4h.values
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # Align 4h RSI and weekly EMA to 6h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_values)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI turns bearish OR trend turns bearish
            if (rsi_4h_aligned[i] < 50 or 
                close[i] <= ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI turns bullish OR trend turns bullish
            if (rsi_4h_aligned[i] > 50 or 
                close[i] >= ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: RSI oversold with uptrend and volume
            if (rsi_4h_aligned[i] < 30 and 
                close[i] > ema_50_1w_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: RSI overbought with downtrend and volume
            elif (rsi_4h_aligned[i] > 70 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals