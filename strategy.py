#!/usr/bin/env python3
# 6h_Volatility_Skew_Rebound
# Hypothesis: During high volatility periods, price tends to revert to the mean after extreme moves.
# Uses a volatility-adjusted RSI on 6h timeframe with 12h trend filter.
# Long when volatility is high, RSI is oversold, and 12h trend is up.
# Short when volatility is high, RSI is overbought, and 12h trend is down.
# Volatility filter prevents trading in low-vol chop where mean reversion fails.
# Works in bull/bear by following 12h trend direction while capturing mean reversion in high vol.

name = "6h_Volatility_Skew_Rebound"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA34 for trend direction
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volatility: ATR(6) on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr6 = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    
    # Volatility ratio: current ATR6 vs 24-period average (~4 days)
    atr6_ma_24 = pd.Series(atr6).rolling(window=24, min_periods=24).mean().values
    volatility_ratio = atr6 / atr6_ma_24
    high_vol = volatility_ratio > 1.5  # Volatility spike
    
    # RSI(6) on 6h close
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # RSI extremes
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(atr6[i]) or 
            np.isnan(atr6_ma_24[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: High vol + RSI oversold + 12h uptrend
            if high_vol[i] and rsi_oversold[i] and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: High vol + RSI overbought + 12h downtrend
            elif high_vol[i] and rsi_overbought[i] and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Volatility drops or RSI neutral or trend breaks
            if not high_vol[i] or rsi[i] >= 50 or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Volatility drops or RSI neutral or trend breaks
            if not high_vol[i] or rsi[i] <= 50 or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals