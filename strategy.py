#!/usr/bin/env python3
"""
1d_1w_Keltner_RSI_Strategy
Hypothesis: On the daily timeframe, price touching the lower Keltner Band with RSI oversold signals mean reversion in ranging markets, while touching the upper band with RSI overbought signals continuation in trending markets. Uses 1-week trend filter to align with higher timeframe momentum. Low trade frequency (<20/year) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Keltner_RSI_Strategy"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === DAILY INDICATORS ===
    # Keltner Channel (20, 2)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(rsi[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price at lower Keltner + RSI oversold + weekly uptrend
        long_signal = (close[i] <= lower_keltner[i]) and (rsi[i] < 30) and (close[i] > ema50_1w_aligned[i])
        
        # Short: price at upper Keltner + RSI overbought + weekly downtrend
        short_signal = (close[i] >= upper_keltner[i]) and (rsi[i] > 70) and (close[i] < ema50_1w_aligned[i])
        
        # Exit: RSI returns to neutral zone (40-60)
        exit_signal = (rsi[i] >= 40) and (rsi[i] <= 60)
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_signal and position != 0:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals