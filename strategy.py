#!/usr/bin/env python3
"""
6h_1w_MultiFactor_Position_Sizing
Hypothesis: Combines weekly trend filter (EMA50), 6h momentum (RSI divergence), and volatility expansion (ATR ratio) for high-probability entries. 
Weekly EMA50 defines structural trend, RSI divergence identifies exhaustion points, ATR ratio >1.5 confirms momentum breakout. 
Works in bull markets via trend continuation and in bear markets via counter-trend reversals at key levels. 
Target: 15-25 trades/year per symbol with disciplined position sizing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 60-period ATR for volatility measurement
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=60, min_periods=60).mean().values
    atr_ma = pd.Series(atr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr / atr_ma  # Current volatility vs recent average
    
    # 6-period RSI for momentum exhaustion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        ema50_1w = np.full(len(prices), np.nan)
    else:
        close_1w = df_1w['close'].values
        ema50_1w_raw = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema50_1w = align_htf_to_ltf(prices, df_1w, ema50_1w_raw)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(ema50_1w[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: weekly uptrend + RSI oversold + volatility expansion
        long_condition = (close[i] > ema50_1w[i] and 
                         rsi[i] < 30 and 
                         atr_ratio[i] > 1.5)
        
        # Short conditions: weekly downtrend + RSI overbought + volatility expansion
        short_condition = (close[i] < ema50_1w[i] and 
                          rsi[i] > 70 and 
                          atr_ratio[i] > 1.5)
        
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

name = "6h_1w_MultiFactor_Position_Sizing"
timeframe = "6h"
leverage = 1.0