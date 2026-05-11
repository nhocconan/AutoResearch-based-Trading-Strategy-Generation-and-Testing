#1d_WeeklyTrend_Reversal_With_Filter
# Weekly trend direction with daily reversal signals filtered by volume and volatility
# Designed to work in both bull and bear markets by following weekly momentum
# while entering on daily pullbacks with confirmation filters to reduce overtrading
# Target: 15-25 trades/year per symbol

#!/usr/bin/env python3
name = "1d_WeeklyTrend_Reversal_With_Filter"
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
    
    # Weekly trend: price above/below weekly EMA20
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    weekly_uptrend = close > ema_1w_aligned
    
    # Daily RSI(14) for reversal signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    # Volatility filter: avoid extremely low volatility days
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - np.roll(low, 1))
    tr3 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    vol_filter = atr > (pd.Series(atr).rolling(window=50, min_periods=50).mean().values * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend + RSI oversold reversal + volume + volatility filter
            if (weekly_uptrend[i] and 
                rsi[i] < 30 and rsi[i-1] >= 30 and 
                volume_filter[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + RSI overbought reversal + volume + volatility filter
            elif ((not weekly_uptrend[i]) and 
                  rsi[i] > 70 and rsi[i-1] <= 70 and 
                  volume_filter[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down OR RSI overbought
            if (not weekly_uptrend[i]) or (rsi[i] > 70 and rsi[i-1] <= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up OR RSI oversold
            if weekly_uptrend[i] or (rsi[i] < 30 and rsi[i-1] >= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals