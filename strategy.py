#!/usr/bin/env python3
"""
1h_RSI_MeanReversion_With_4hTrend_Filter
Hypothesis: In strong 4h trends, RSI mean reversion on 1h captures pullbacks with high win rate.
Long when 4h EMA50 up + RSI(14) < 30; Short when 4h EMA50 down + RSI(14) > 70.
Session filter (08-20 UTC) reduces noise. Fixed size 0.20 limits drawdown.
Works in bull/bear by following 4h trend direction.
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
    
    # 4h EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(14, 50)  # RSI and EMA warmup
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema_trend = ema_4h_aligned[i]
        
        if position == 0:
            # Long: uptrend + oversold
            if ema_trend > close[i] and rsi_val < 30:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + overbought
            elif ema_trend < close[i] and rsi_val > 70:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or trend change
            if rsi_val > 70 or ema_trend < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI oversold or trend change
            if rsi_val < 30 or ema_trend > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_With_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0