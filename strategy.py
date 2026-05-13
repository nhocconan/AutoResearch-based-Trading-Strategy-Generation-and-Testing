#!/usr/bin/env python3
"""
4h_RSI_DivMACD_TrendFilter
Hypothesis: RSI divergence signals momentum exhaustion, while MACD histogram confirms trend strength.
Long on bullish RSI divergence + positive MACD histogram + price above 4h EMA50.
Short on bearish RSI divergence + negative MACD histogram + price below 4h EMA50.
Uses 12h EMA50 as higher timeframe trend filter to avoid counter-trend trades.
Designed for low trade frequency (<30/year) to minimize fee drag in ranging markets.
"""

name = "4h_RSI_DivMACD_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI(14) with proper min_periods
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # MACD(12,26,9)
    ema_fast = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - macd_signal
    macd_hist = macd_hist.values
    
    # 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h EMA50 as higher timeframe trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = df_12h['close'].values > ema_50_12h
    downtrend_12h = df_12h['close'].values < ema_50_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # RSI divergence detection (bullish: price makes LL, RSI makes HL)
    # Bearish: price makes HH, RSI makes LH
    rsi_bull_div = np.zeros(n, dtype=bool)
    rsi_bear_div = np.zeros(n, dtype=bool)
    
    # Look for divergences over 3-bar windows to reduce noise
    for i in range(10, n):
        # Bullish divergence: price lower low, RSI higher low
        if (low[i] < low[i-3] and low[i] < low[i-6] and 
            rsi[i] > rsi[i-3] and rsi[i] > rsi[i-6]):
            rsi_bull_div[i] = True
        # Bearish divergence: price higher high, RSI lower high
        if (high[i] > high[i-3] and high[i] > high[i-6] and 
            rsi[i] < rsi[i-3] and rsi[i] < rsi[i-6]):
            rsi_bear_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: bullish RSI divergence + positive MACD histogram + price above EMA50 + 12h uptrend
            if (rsi_bull_div[i] and macd_hist[i] > 0 and 
                close[i] > ema_50[i] and uptrend_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: bearish RSI divergence + negative MACD histogram + price below EMA50 + 12h downtrend
            elif (rsi_bear_div[i] and macd_hist[i] < 0 and 
                  close[i] < ema_50[i] and downtrend_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish RSI divergence or MACD histogram turns negative
            if rsi_bear_div[i] or macd_hist[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish RSI divergence or MACD histogram turns positive
            if rsi_bull_div[i] or macd_hist[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals