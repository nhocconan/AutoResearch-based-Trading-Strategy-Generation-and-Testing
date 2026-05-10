#!/usr/bin/env python3
"""
1h_RSI_Extreme_4hTrend_Filter
Hypothesis: In 1h timeframe, take long when RSI(14) < 30 and 4h EMA50 trend is up,
take short when RSI(14) > 70 and 4h EMA50 trend is down. Uses 4h EMA50 for trend direction
to avoid counter-trend trades. Position size fixed at 0.20 to limit drawdown.
Designed for low trade frequency (15-35/year) by requiring both RSI extreme and trend alignment.
Works in bull markets (catching dips in uptrend) and bear markets (selling rallies in downtrend).
"""

name = "1h_RSI_Extreme_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # RSI needs 14 periods
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold AND price above 4h EMA50 (uptrend)
            if rsi[i] < 30 and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought AND price below 4h EMA50 (downtrend)
            elif rsi[i] > 70 and close[i] < ema_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral (50) or breaks below 4h EMA50
            if rsi[i] >= 50 or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI returns to neutral (50) or breaks above 4h EMA50
            if rsi[i] <= 50 or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals