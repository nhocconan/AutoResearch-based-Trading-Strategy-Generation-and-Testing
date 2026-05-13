#!/usr/bin/env python3
"""
1d_RSI200_Pullback_Trend_Filter
Hypothesis: Buy pullbacks to EMA200 in strong uptrends and sell pullbacks to EMA200 in strong downtrends.
Uptrend defined as price > EMA200 and RSI > 50; downtrend as price < EMA200 and RSI < 50.
Enter long when price crosses above EMA200 in uptrend, short when crosses below EMA200 in downtrend.
Use weekly EMA200 as higher timeframe trend filter to avoid counter-trend trades.
Target: 10-25 trades per year per symbol.
"""

name = "1d_RSI200_Pullback_Trend_Filter"
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
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA200 on daily
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Weekly EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    uptrend_1w = df_1w['close'].values > ema_200_1w
    downtrend_1w = df_1w['close'].values < ema_200_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        price = close[i]
        ema200 = ema_200[i]
        rsi_val = rsi[i]
        uptrend_weekly = uptrend_1w_aligned[i]
        downtrend_weekly = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: price crosses above EMA200, in uptrend (price > EMA200 and RSI > 50), weekly uptrend
            if price > ema200 and rsi_val > 50 and uptrend_weekly:
                # Check for crossover: previous price was at or below EMA200
                if i > 0 and close[i-1] <= ema_200[i-1]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            # SHORT: price crosses below EMA200, in downtrend (price < EMA200 and RSI < 50), weekly downtrend
            elif price < ema200 and rsi_val < 50 and downtrend_weekly:
                # Check for crossover: previous price was at or above EMA200
                if i > 0 and close[i-1] >= ema_200[i-1]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below EMA200 or RSI < 40
            if price < ema200 or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above EMA200 or RSI > 60
            if price > ema200 or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals