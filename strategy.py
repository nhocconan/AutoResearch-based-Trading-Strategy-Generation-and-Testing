#!/usr/bin/env python3
"""
1h_RSI2_Pullback_4hTrend
Hypothesis: Buy pullbacks in 4h uptrends using RSI(2) on 1h, sell rallies in 4h downtrends.
- 4h trend: EMA50 > EMA200 for uptrend, EMA50 < EMA200 for downtrend
- 1h entry: RSI(2) < 10 for long in uptrend, RSI(2) > 90 for short in downtrend
- Exit: RSI(2) > 50 for long exit, RSI(2) < 50 for short exit
- Uses extreme short-term RSI to catch mean reversion within strong trends
- Designed for low trade frequency (~20-40/year) with high win rate in trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h trend filter: EMA50 vs EMA200
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(df_4h['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # 1h RSI(2)
    rsi_period = 2
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, rsi_period+1) + 1
    
    for i in range(start_idx, n):
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h uptrend + RSI(2) oversold
            if ema50_4h_aligned[i] > ema200_4h_aligned[i] and rsi[i] < 10:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + RSI(2) overbought
            elif ema50_4h_aligned[i] < ema200_4h_aligned[i] and rsi[i] > 90:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI(2) > 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI(2) < 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_Pullback_4hTrend"
timeframe = "1h"
leverage = 1.0