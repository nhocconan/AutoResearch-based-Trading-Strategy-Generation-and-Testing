#!/usr/bin/env python3
"""
4h_1d_RSI_MeanReversion_With_Trend_Filter
Hypothesis: In BTC/ETH, RSI extremes combined with 4h EMA50 trend filter provide mean-reversion entries that work in both bull and bear markets. 
Long when RSI(14) < 30 and price > EMA50 (bullish bias in range). 
Short when RSI(14) > 70 and price < EMA50 (bearish bias in range). 
Exit when RSI returns to neutral (40-60 range) or trend filter fails.
Uses 1d timeframe only for context (no calculation), focusing on 4h RSI and EMA.
Designed for low trade frequency (~20-30/year) with high win rate via confluence.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h RSI(14) with proper min_periods
    close = prices['close']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4h EMA50 for trend filter
    ema_50 = close.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(rsi[i]) or np.isnan(ema_50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close.iloc[i]
        rsi_val = rsi[i]
        ema_val = ema_50[i]
        
        if position == 0:
            # Long: RSI oversold + price above EMA50 (bullish bias)
            if rsi_val < 30 and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + price below EMA50 (bearish bias)
            elif rsi_val > 70 and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (40-60) OR price breaks below EMA50
            if rsi_val > 40 or price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (40-60) OR price breaks above EMA50
            if rsi_val < 60 or price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_RSI_MeanReversion_With_Trend_Filter"
timeframe = "4h"
leverage = 1.0