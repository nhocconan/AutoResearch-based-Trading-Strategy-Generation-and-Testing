#!/usr/bin/env python3
"""
4h_RSI_40_60_Pullback_12hTrend_Filter
Hypothesis: On 4h timeframe, enter long when RSI(14) pulls back to 40-50 in a 12h uptrend,
and short when RSI pulls back to 50-60 in a 12h downtrend. Uses 12h EMA50 as trend filter.
Designed for mean reversion within trend, works in bull/bear by only trading with trend.
Low frequency: targets 20-40 trades/year per symbol.
"""
name = "4h_RSI_40_60_Pullback_12hTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend
    close_12h = pd.Series(df_12h['close'])
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI pulls back to 40-50 in 12h uptrend
            if (40 <= rsi[i] <= 50 and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI pulls back to 50-60 in 12h downtrend
            elif (50 <= rsi[i] <= 60 and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: RSI reaches opposite extreme or trend change
            if position == 1:
                if rsi[i] >= 60 or close[i] < ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
            else:  # position == -1
                if rsi[i] <= 40 or close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
            # Hold position
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals