#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_Chop_Filter
# Hypothesis: KAMA adapts to market efficiency, reducing lag in trends and whipsaw in ranges. 
# Combined with RSI for momentum and Choppiness index for regime detection, this filters false signals.
# Long when KAMA up, RSI > 50, and Chop < 38.2 (trending). Short when KAMA down, RSI < 50, and Chop < 38.2.
# Chop > 61.8 triggers range mode: mean reversion at Bollinger Bands (20,2).
# Works in bull markets (trend follow) and bear markets (range mean reversion).
# Target: 12-37 trades/year per symbol to minimize fee drag.

name = "12h_KAMA_Trend_RSI_Chop_Filter"
timeframe = "12h"
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

    # Get weekly data for Chop filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index (14)
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros(len(close))
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first TR is just high-low
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        
        max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        chop = np.zeros(len(close))
        for i in range(len(close)):
            if atr[i] > 0 and (max_high[i] - min_low[i]) > 0:
                chop[i] = 100 * np.log10(atr[i] * period / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50.0
        return chop
    
    chop = choppiness_index(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # KAMA (10,2,30)
    def kama(close, er_period=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > er_period else np.zeros(len(close)-er_period+1)
        # Vectorized volatility calculation
        volatility = np.array([np.sum(np.abs(np.diff(close[i:i+er_period]))) for i in range(len(close)-er_period+1)])
        er = np.zeros(len(close))
        er[er_period-1:] = change / (volatility + 1e-10)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros(len(close))
        kama[er_period-1] = close[er_period-1]
        for i in range(er_period, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        # Pad beginning
        kama[:er_period-1] = kama[er_period-1]
        return kama
    
    kama_val = kama(close, 10, 2, 30)
    
    # RSI (14)
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        rsi_val[:period] = 50.0  # neutral before enough data
        return rsi_val
    
    rsi_val = rsi(close, 14)
    
    # Bollinger Bands (20,2) for mean reversion in ranging markets
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(kama_val[i]) or 
            np.isnan(rsi_val[i]) or 
            np.isnan(upper[i]) or 
            np.isnan(lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        chop_val = chop_aligned[i]
        kama_now = kama_val[i]
        kama_prev = kama_val[i-1]
        rsi_now = rsi_val[i]
        price = close[i]

        if position == 0:
            # Trending regime: Chop < 38.2
            if chop_val < 38.2:
                # KAMA turning up + RSI > 50
                if kama_now > kama_prev and rsi_now > 50:
                    signals[i] = 0.25
                    position = 1
                # KAMA turning down + RSI < 50
                elif kama_now < kama_prev and rsi_now < 50:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Ranging regime: Chop > 61.8
            elif chop_val > 61.8:
                # Mean reversion at Bollinger Bands
                if price <= lower[i] and rsi_now < 30:  # oversold
                    signals[i] = 0.25
                    position = 1
                elif price >= upper[i] and rsi_now > 70:  # overbought
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition zone: no action
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: trend change or range entry
            if chop_val > 61.8:  # switched to range, exit trend follow
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2:  # still trending
                if kama_now < kama_prev or rsi_now < 50:  # trend weakening
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: trend change or range entry
            if chop_val > 61.8:  # switched to range, exit trend follow
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2:  # still trending
                if kama_now > kama_prev or rsi_now > 50:  # trend weakening
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25

    return signals