#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend direction + RSI(2) extreme + volume spike (>1.5x 20-bar avg) for mean reversion entries in choppy markets. Uses 1w EMA50 as regime filter (only trade when price > EMA50 in bull, < EMA50 in bear). Designed for BTC/ETH robustness: KAMA adapts to market noise, RSI(2) captures short-term exhaustion, volume confirms conviction, and 1w EMA50 avoids counter-trend trades. Targets 7-25 trades/year on 1d timeframe.

name = "1d_KAMA_Trend_RSI2_VolumeSpike_1wEMA50_Regime_v1"
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
    volume = prices['volume'].values
    
    # Calculate 1d KAMA (adaptive trend)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix array lengths
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1w EMA50 for regime filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate RSI(2) for mean reversion
    delta = np.diff(close)
    delta = np.concatenate([np.full(1, np.nan), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=2, min_periods=2).mean().values
    avg_loss = pd.Series(loss).rolling(window=2, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)  # neutral when undefined
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Regime filter: only long when price > 1w EMA50, short when price < 1w EMA50
            if close[i] > ema_50_1w_aligned[i]:
                # LONG: KAMA up (trend), RSI(2) oversold (<10), volume spike (>1.5x avg)
                if (kama[i] > kama[i-1] and 
                    rsi[i] < 10 and 
                    volume[i] > 1.5 * avg_volume[i]):
                    signals[i] = 0.25
                    position = 1
            else:  # price < 1w EMA50 (bear regime)
                # SHORT: KAMA down (trend), RSI(2) overbought (>90), volume spike (>1.5x avg)
                if (kama[i] < kama[i-1] and 
                    rsi[i] > 90 and 
                    volume[i] > 1.5 * avg_volume[i]):
                    signals[i] = -0.25
                    position = -1
            signals[i] = 0.0  # explicit flat
        elif position == 1:
            # EXIT LONG: RSI(2) becomes overbought (>80) OR price crosses below KAMA (trend change)
            if (rsi[i] > 80 or 
                close[i] < kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI(2) becomes oversold (<20) OR price crosses above KAMA (trend change)
            if (rsi[i] < 20 or 
                close[i] > kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals