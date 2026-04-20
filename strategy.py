#!/usr/bin/env python3
"""
4h_RSI34_MeanReversion_1dTrend_v1
Concept: 4h RSI mean reversion with daily trend filter.
- Long: RSI(34) < 30 AND daily close > EMA(50)
- Short: RSI(34) > 70 AND daily close < EMA(50)
- Exit: RSI crosses back to 50 (neutral)
- Position sizing: 0.25
- Target: 75-200 total trades over 4 years
- Works in bull/bear: Daily EMA filter ensures trades align with higher timeframe trend, RSI captures short-term mean reversion
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_RSI34_MeanReversion_1dTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === Daily: EMA Trend Filter (50-period) ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h: RSI(34) ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/34, adjust=False, min_periods=34).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/34, adjust=False, min_periods=34).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for RSI
    
    for i in range(start_idx, n):
        # Get values
        rsi_val = rsi[i]
        ema50 = ema_50_1d_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(rsi_val) or np.isnan(ema50):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold in uptrend
            if rsi_val < 30 and ema50 > prices['close'].values[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in downtrend
            elif rsi_val > 70 and ema50 < prices['close'].values[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (50)
            if rsi_val >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (50)
            if rsi_val <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals