#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_Filter_Trend_200
# Hypothesis: KAMA adapts to market noise, capturing trend direction while avoiding whipsaws. Combined with RSI(14) for overbought/oversold conditions and a 200-period EMA for long-term trend filter, this strategy aims to capture sustained moves in both bull and bear markets. The KAMA provides adaptive trend direction, RSI filters extremes, and EMA200 ensures alignment with the dominant trend. Works in bull markets (buy when KAMA up, RSI not overbought, price > EMA200) and bear markets (sell when KAMA down, RSI not oversold, price < EMA200).

name = "4h_KAMA_Direction_RSI_Filter_Trend_200"
timeframe = "4h"
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

    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily timeframe
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)

    # Calculate KAMA on 4h timeframe
    def kama(close, window=10, pow1=2, pow2=30):
        """Kaufman Adaptive Moving Average"""
        change = np.abs(np.diff(close, n=window))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[window:] = change[window-1:] / volatility[window-1:]
        sc = (er * (2/(pow1+1) - 2/(pow2+1)) + 2/(pow2+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    kama_values = kama(close, window=10, pow1=2, pow2=30)
    
    # Calculate RSI(14) on 4h timeframe
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_values = rsi(close, period=14)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(kama_values[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising (trend up), RSI not overbought, price above EMA200
            if (kama_values[i] > kama_values[i-1] and 
                rsi_values[i] < 70 and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (trend down), RSI not oversold, price below EMA200
            elif (kama_values[i] < kama_values[i-1] and 
                  rsi_values[i] > 30 and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling or RSI overbought
            if (kama_values[i] < kama_values[i-1] or rsi_values[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising or RSI oversold
            if (kama_values[i] > kama_values[i-1] or rsi_values[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals