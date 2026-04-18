# 2025-01-13
# 2025-01-13
#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d ATR-based volatility filter + 1w EMA trend filter.
Buy when volatility is low (ATR < 20-period SMA) and price crosses above 1w EMA(50).
Sell when volatility is high (ATR > 20-period SMA) or price crosses below 1w EMA(50).
Volatility filter prevents whipsaws in choppy markets, EMA trend filter ensures directional bias.
Designed for ~20 trades/year to minimize fee drag. Works in bull (trend following) and bear (mean reversion in low vol).
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
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on 1d
    tr = np.zeros(len(close_1d))
    atr = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(hl, hc, lc)
    
    # ATR(14) calculation
    if len(tr) >= 14:
        atr[13] = np.mean(tr[1:15])  # First ATR value
        for i in range(15, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # ATR 20-period SMA for volatility regime
    atr_sma = np.full(len(close_1d), np.nan)
    if len(atr) >= 20:
        for i in range(19, len(atr)):
            atr_sma[i] = np.mean(atr[i-19:i+1])
    
    # Get 1w data for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Align indicators to 4h timeframe
    atr_sma_4h = align_htf_to_ltf(prices, df_1d, atr_sma)
    atr_4h = align_htf_to_ltf(prices, df_1d, atr)
    ema_50_1w_4h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need EMA and ATR data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1w_4h[i]) or np.isnan(atr_sma_4h[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: low vol when ATR < SMA(ATR)
        low_volatility = atr_4h[i] < atr_sma_4h[i]
        high_volatility = atr_4h[i] > atr_sma_4h[i]
        
        # Trend filter: price relative to 1w EMA50
        price_above_ema = close[i] > ema_50_1w_4h[i]
        price_below_ema = close[i] < ema_50_1w_4h[i]
        
        if position == 0:
            # Long entry: low volatility + price above 1w EMA50
            if low_volatility and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: high volatility + price below 1w EMA50
            elif high_volatility and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: high volatility or price below EMA
            if high_volatility or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: low volatility or price above EMA
            if low_volatility or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ATR_Volatility_Regime_1wEMA50"
timeframe = "4h"
leverage = 1.0