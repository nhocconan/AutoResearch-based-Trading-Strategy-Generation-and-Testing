#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend direction with RSI(2) mean reversion entries and 1w EMA34 trend filter.
# Uses 1w EMA34 for higher timeframe trend alignment (HTF), 1d KAMA for primary trend direction,
# and RSI(2) < 10 for long entries / > 90 for short entries during pullbacks.
# Designed for low trade frequency (target 30-100 total over 4 years) to minimize fee drag.
# Works in bull markets by following the 1w trend and buying dips, and in bear markets
# by selling rallies against the trend. Volume confirmation is not used to avoid overtrading.

name = "1d_KAMA_RSI2_1wEMA34_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d KAMA (ER=10) for primary trend
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(2) for mean reversion entries
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback for stability
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(kama[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1w uptrend (price > EMA34) + 1d KAMA uptrend (price > KAMA) + RSI(2) oversold (<10)
            if (close[i] > ema_34_1w_aligned[i] and 
                close[i] > kama[i] and 
                rsi[i] < 10):
                signals[i] = 0.25
                position = 1
            # SHORT: 1w downtrend (price < EMA34) + 1d KAMA downtrend (price < KAMA) + RSI(2) overbought (>90)
            elif (close[i] < ema_34_1w_aligned[i] and 
                  close[i] < kama[i] and 
                  rsi[i] > 90):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI(2) overbought (>80) or trend breaks (price < KAMA)
            if (rsi[i] > 80) or (close[i] < kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI(2) oversold (<20) or trend breaks (price > KAMA)
            if (rsi[i] < 20) or (close[i] > kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals