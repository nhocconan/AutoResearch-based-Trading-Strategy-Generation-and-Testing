#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h KAMA direction + RSI + Choppiness filter
    # KAMA adapts to market noise - follows trend in trending markets, stays flat in chop
    # RSI(14) for momentum confirmation with extreme levels
    # Choppiness Index (14) to filter: >61.8 = range (mean revert), <38.2 = trend (trend follow)
    # Works in bull/bear: KAMA avoids whipsaws in chop, RSI captures momentum in trends
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA calculation (adaptive moving average)
    def calculate_kama(price, er_period=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(price, prepend=price[0]))
        volatility = np.abs(np.diff(price)).cumsum() - np.abs(np.diff(price, prepend=price[0])).cumsum()
        volatility = np.where(volatility == 0, 1, volatility)
        er = change / volatility
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(price)
        kama[0] = price[0]
        for i in range(1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    # Choppiness Index calculation
    def calculate_chop(high, low, close, period=14):
        atr = np.maximum(high - low, 
                         np.maximum(np.abs(high - np.roll(close, 1)),
                                    np.abs(low - np.roll(close, 1))))
        atr[0] = high[0] - low[0]
        sum_atr = np.nancumsum(atr) - np.nancumsum(np.roll(atr, period))  # rolling sum
        sum_atr[:period] = np.nancumsum(atr[:period+1])
        max_range = np.maximum.accumulate(high) - np.minimum.accumulate(low)
        chop = 100 * np.log10(sum_atr / period / max_range) / np.log10(period)
        return chop
    
    # KAMA on close
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index
    chop = calculate_chop(high, low, close, period=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA (uptrend) AND RSI > 50 (bullish momentum) AND chop < 38.2 (trending market)
            if close[i] > kama[i] and rsi[i] > 50 and chop[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA (downtrend) AND RSI < 50 (bearish momentum) AND chop < 38.2 (trending market)
            elif close[i] < kama[i] and rsi[i] < 50 and chop[i] < 38.2:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses KAMA in opposite direction OR chop > 61.8 (market becomes choppy)
            if position == 1:
                if close[i] < kama[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_Chop_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0