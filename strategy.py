#!/usr/bin/env python3
"""
1h RSI + 4h EMA Trend + Volume Confirmation
Hypothesis: In strong trends (4h EMA), 1h RSI pullbacks offer high-probability entries.
Volume confirms institutional participation. Works in bull (buy dips in uptrend) and bear
(sell rallies in downtrend). Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14394_1h_rsi_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for EMA trend (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) for trend filter
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h Volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = 50  # warmup for EMA and RSI
    
    for i in range(start, n):
        if np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long
            if (rsi[i] > 70 or  # overbought exit
                close[i] <= entry_price - 1.5 * (high[i] - low[i]) or  # simple volatility stop
                ema_4h_aligned[i] < ema_4h_aligned[i-1]):  # trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short
            if (rsi[i] < 30 or  # oversold exit
                close[i] >= entry_price + 1.5 * (high[i] - low[i]) or
                ema_4h_aligned[i] > ema_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Entry conditions: RSI extreme + volume + 4h trend alignment
            vol_filter = volume[i] > 1.2 * vol_ma[i]  # above average volume
            
            long_setup = (rsi[i] < 30 and  # oversold
                         close[i] > ema_4h_aligned[i] and  # above 4h EMA (uptrend)
                         vol_filter)
            
            short_setup = (rsi[i] > 70 and  # overbought
                          close[i] < ema_4h_aligned[i] and  # below 4h EMA (downtrend)
                          vol_filter)
            
            if long_setup:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals