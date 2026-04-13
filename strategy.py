#!/usr/bin/env python3
"""
6h_12h_1d_RSI_Confluence_V2
Hypothesis: Long when price > 12h EMA20 and 1d RSI < 30 (oversold), short when price < 12h EMA20 and 1d RSI > 70 (overbought).
Uses 12h EMA for trend filter and 1d RSI for mean reversion extremes. Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend).
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 12h data for EMA20
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        if np.isnan(ema_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long: price above 12h EMA20 AND 1d RSI oversold (<30)
        long_condition = (close[i] > ema_12h_aligned[i]) and (rsi_1d_aligned[i] < 30)
        
        # Short: price below 12h EMA20 AND 1d RSI overbought (>70)
        short_condition = (close[i] < ema_12h_aligned[i]) and (rsi_1d_aligned[i] > 70)
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_12h_1d_RSI_Confluence_V2"
timeframe = "6h"
leverage = 1.0