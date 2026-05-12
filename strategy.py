#!/usr/bin/env python3
# 1D_KAMA_REVERSAL_FILTER
# Hypothesis: KAMA identifies trend direction, RSI provides mean-reversion signals, and chop filter (ADX) avoids trending markets.
# Long when KAMA slope positive, RSI < 30, and ADX < 25 (range market). Short when KAMA slope negative, RSI > 70, and ADX < 25.
# Exit when RSI crosses 50 or trend changes. Designed to capture reversals in range-bound markets while avoiding whipsaws in trends.
# Targets 15-25 trades/year to minimize fee drain with high-probability setups.

name = "1D_KAMA_REVERSAL_FILTER"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # KAMA (10-period ER, 2 and 30 SC)
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    volatility = np.concatenate([[np.sum(np.abs(np.diff(close[:11])))], volatility[10:]])
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # ADX (14-period)
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Weekly trend filter (1w EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    pclose_w = df_1w['close'].values
    ema1w = pd.Series(pclose_w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1w_aligned = align_htf_to_ltf(prices, df_1w, ema1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(ema1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # KAMA slope (direction)
        kama_slope = kama[i] - kama[i-1]
        
        if position == 0:
            # LONG: KAMA up, RSI oversold, low ADX (range), above weekly trend
            if kama_slope > 0 and rsi[i] < 30 and adx[i] < 25 and close[i] > ema1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down, RSI overbought, low ADX (range), below weekly trend
            elif kama_slope < 0 and rsi[i] > 70 and adx[i] < 25 and close[i] < ema1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses 50 or KAMA turns down
            if rsi[i] > 50 or kama_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses 50 or KAMA turns up
            if rsi[i] < 50 or kama_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals