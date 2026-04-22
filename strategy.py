#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA + RSI(14) + Chop Filter with 1d Trend Filter
# KAMA adapts to market noise - faster in trends, slower in ranges
# RSI(14) provides momentum confirmation (long when RSI>50, short when RSI<50)
# Chop filter (Choppiness Index) avoids whipsaws in ranging markets (CHOP>61.8)
# 1d EMA(34) trend filter ensures trades align with higher timeframe trend
# Designed for 4h timeframe targeting 20-35 trades/year per symbol
# Should work in both bull (trend following) and bear (mean reversion in ranges)

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
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) on 4h data
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Calculate ER over 10 periods
    er = np.zeros(n)
    for i in range(10, n):
        if volatility[i] != 0:
            er[i] = np.sum(change[i-9:i+1]) / volatility[i]
        else:
            er[i] = 0
    # SC = [ER * (fastest - slowest) + slowest]^2
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) on 4h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (CHOP) on 4h data - avoids ranging markets
    atr_period = 14
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = np.zeros(n)
    for i in range(atr_period, n):
        if highest_high[i] != lowest_low[i]:
            chop[i] = 100 * np.log10(np.sum(tr[i-atr_period+1:i+1]) / np.log(atr_period) / (highest_high[i] - lowest_low[i]))
        else:
            chop[i] = 50
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA + RSI > 50 + CHOP < 61.8 (trending) + 1d uptrend
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                chop[i] < 61.8 and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA + RSI < 50 + CHOP < 61.8 (trending) + 1d downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8 and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses KAMA OR trend reversal OR chop too high (ranging)
            if position == 1:
                if (close[i] < kama[i] or 
                    close[i] < ema_34_1d_aligned[i] or 
                    chop[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > kama[i] or 
                    close[i] > ema_34_1d_aligned[i] or 
                    chop[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_ChopFilter_1dEMA34_Trend"
timeframe = "4h"
leverage = 1.0