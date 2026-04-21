#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d breakout above 20-day high with weekly RSI(7) oversold filter.
# Long when price breaks above 20-day high and weekly RSI < 30 (oversold bounce).
# Exit when price breaks below 10-day low or weekly RSI > 70 (overbought).
# Uses weekly RSI to catch mean-reversion bounces in both bull and bear markets.
# Target: 10-20 trades/year by requiring breakout + oversold conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly RSI(7)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    avg_gain = wilder_smooth(gain, 7)
    avg_loss = wilder_smooth(loss, 7)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Calculate daily 20-period high and 10-period low
    high_20 = prices['high'].rolling(window=20, min_periods=20).max()
    low_10 = prices['low'].rolling(window=10, min_periods=10).min()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(rsi_aligned[i]) or np.isnan(high_20.iloc[i]) or np.isnan(low_10.iloc[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Enter long: price breaks above 20-day high AND weekly RSI < 30 (oversold)
            if price > high_20.iloc[i] and rsi_aligned[i] < 30:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit: price breaks below 10-day low OR weekly RSI > 70 (overbought)
            if price < low_10.iloc[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "1d_Breakout20_High_WRSI7_Oversold"
timeframe = "1d"
leverage = 1.0