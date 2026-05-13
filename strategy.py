#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Chop_Filter
# Hypothesis: KAMA identifies trend direction with low lag, RSI filters overbought/oversold, and Choppiness Index identifies ranging markets for mean reversion.
# Works in bull markets (trend following) and bear markets (mean reversion in ranges).
# Target: 15-25 trades/year per symbol to minimize fee drag.

name = "1d_KAMA_Trend_RSI_Chop_Filter"
timeframe = "1d"
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

    # KAMA calculation
    def calculate_kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        for i in range(length, len(close)):
            if volatility[i] != 0:
                er[i] = np.sum(change[i-length+1:i+1]) / volatility[i]
            else:
                er[i] = 0
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    # RSI calculation
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        for i in range(1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Choppiness Index calculation
    def calculate_chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high - low)
        tr2 = np.abs(np.roll(high, 1) - close)
        tr3 = np.abs(np.roll(low, 1) - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        for i in range(1, len(close)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        highest_high = np.maximum.accumulate(high)
        lowest_low = np.minimum.accumulate(low)
        range_hl = highest_high - lowest_low
        chop = 100 * np.log10(np.sum(atr, axis=0) / range_hl) / np.log10(length)
        return chop

    kama = calculate_kama(close, 10, 2, 30)
    rsi = calculate_rsi(close, 14)
    chop = calculate_chop(high, low, close, 14)

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(sma50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Trending market: CHOP < 38.2
            if chop[i] < 38.2:
                # LONG: Price > KAMA and RSI < 70 and weekly uptrend
                if close[i] > kama[i] and rsi[i] < 70 and close[i] > sma50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Price < KAMA and RSI > 30 and weekly downtrend
                elif close[i] < kama[i] and rsi[i] > 30 and close[i] < sma50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Ranging market: CHOP > 61.8
            elif chop[i] > 61.8:
                # LONG: RSI < 30 (oversold)
                if rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                # SHORT: RSI > 70 (overbought)
                elif rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend change or overbought
            if chop[i] > 61.8 and rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            elif close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend change or oversold
            if chop[i] > 61.8 and rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            elif close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals