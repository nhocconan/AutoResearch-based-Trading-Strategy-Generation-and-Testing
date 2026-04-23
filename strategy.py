#!/usr/bin/env python3
"""
Hypothesis: 1h session-based mean reversion with 4h trend filter.
- Trade only during 08-20 UTC (liquid session) to avoid low-volume noise
- Use 4h EMA200 as trend filter: long when price > EMA200, short when price < EMA200
- Enter on 1h RSI(14) extremes: long when RSI < 30 (oversold), short when RSI > 70 (overbought)
- Exit on RSI mean reversion: RSI crosses back to 50
- Fixed size 0.20 to manage drawdown
- Designed for 1h timeframe targeting 60-150 trades over 4 years (15-37/year)
- Works in both bull/bear: trend filter ensures we trade with higher timeframe direction,
  RSI mean reversion captures pullbacks within the trend
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
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Load 4h data for EMA200 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Calculate RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if np.isnan(ema200_4h_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema200_val = ema200_4h_aligned[i]
        rsi_val = rsi[i]
        price = close[i]
        
        if position == 0:
            # Long: in uptrend (price > 4h EMA200) and RSI oversold (< 30)
            if price > ema200_val and rsi_val < 30:
                signals[i] = 0.20
                position = 1
            # Short: in downtrend (price < 4h EMA200) and RSI overbought (> 70)
            elif price < ema200_val and rsi_val > 70:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI mean reversion back to 50
            exit_signal = False
            if position == 1 and rsi_val >= 50:
                exit_signal = True
            elif position == -1 and rsi_val <= 50:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Session_RSI_MeanReversion_4hEMA200"
timeframe = "1h"
leverage = 1.0