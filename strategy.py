#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA(10,2,30) + RSI(14) + Chop(14) regime filter
# KAMA adapts to market noise - fast in trends, slow in ranges.
# RSI(14) filters extremes: long when RSI>50, short when RSI<50.
# Chop > 61.8 = range (mean revert), Chop < 38.2 = trending (trend follow).
# Uses daily trend filter from 1d EMA(34) for multi-timeframe alignment.
# Designed for 12h timeframe targeting 15-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # KAMA on close
    def calculate_kama(close, er_len=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(slow+1) - 2/(fast+1)) + 2/(fast+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close)
    
    # RSI(14)
    def calculate_rsi(close, period=14):
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
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close)
    
    # Choppiness Index(14)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], 
                            np.maximum(tr1, np.maximum(tr2, tr3))])
        for i in range(period, len(tr)):
            atr[i] = np.mean(tr[i-period+1:i+1])
        sum_atr = np.sum(atr)
        highest_high = np.max(high)
        lowest_low = np.min(low)
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(period)
        return np.full_like(close, chop)
    
    chop = calculate_chop(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA + RSI > 50 + Chop < 38.2 (trending) + 1d uptrend
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                chop[i] < 38.2 and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA + RSI < 50 + Chop < 38.2 (trending) + 1d downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  chop[i] < 38.2 and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend change or Chop > 61.8 (range) or price crosses KAMA
            if position == 1:
                if (close[i] < kama[i] or 
                    chop[i] > 61.8 or 
                    rsi[i] < 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > kama[i] or 
                    chop[i] > 61.8 or 
                    rsi[i] > 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_Chop_TrendFilter"
timeframe = "12h"
leverage = 1.0