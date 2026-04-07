#!/usr/bin/env python3
"""
4h KAMA with RSI and Chop Filter
Long when KAMA trend is up and RSI crosses above 50 in low volatility (choppy) market
Short when KAMA trend is down and RSI crosses below 50 in low volatility
Exit when KAMA trend reverses or RSI reaches extreme
KAMA adapts to market noise, reducing whipsaws in chop. RSI adds momentum confirmation.
Chop filter avoids strong trends where mean reversion fails.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_kama_rsi_chop_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === KAMA (Kaufman Adaptive Moving Average) ===
    # Parameters
    fast_ema = 2
    slow_ema = 30
    lookback = 10
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=lookback))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # temporary, will fix below
    
    # Proper volatility calculation (sum of absolute changes over lookback period)
    volatility = np.zeros(n)
    for i in range(lookback, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-lookback:i])))
    
    # Avoid division by zero
    er = np.zeros(n)
    for i in range(lookback, n):
        if volatility[i] > 0:
            er[i] = change[i-lookback] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = np.zeros(n)
    fast_sc = 2 / (fast_ema + 1)
    slow_sc = 2 / (slow_ema + 1)
    for i in range(n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) ===
    def compute_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # Initial average
        if len(prices) > period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            # Wilder's smoothing
            for i in range(period+1, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = compute_rsi(close, 14)
    
    # === Chopiness Index (14) ===
    def compute_chop(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.max(tr[:3]) if len(tr) >= 3 else np.max(tr) if len(tr) > 0 else 0], tr])
        
        # ATR
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            if i < period:
                atr[i] = np.mean(tr[:i+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Sum of true range over period
        sum_tr = np.zeros_like(close)
        for i in range(period-1, len(close)):
            sum_tr[i] = np.sum(tr[i-period+1:i+1])
        
        # Max and min close over period
        max_close = np.zeros_like(close)
        min_close = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                max_close[i] = np.max(close[:i+1])
                min_close[i] = np.min(close[:i+1])
            else:
                max_close[i] = np.max(close[i-period+1:i+1])
                min_close[i] = np.min(close[i-period+1:i+1])
        
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if max_close[i] != min_close[i]:
                chop[i] = 100 * np.log10(sum_tr[i] / (max_close[i] - min_close[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop = compute_chop(high, low, close, 14)
    
    # Get 1d trend for filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) > 0:
        # Simple 1d trend: close > SMA(50)
        close_1d = df_1d['close'].values
        sma_50 = np.zeros(len(close_1d))
        for i in range(len(close_1d)):
            if i < 50:
                sma_50[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[i]
            else:
                sma_50[i] = np.mean(close_1d[i-49:i+1])
        trend_1d = close_1d > sma_50
        trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d.astype(float))
    else:
        trend_1d_aligned = np.ones(n)  # neutral if no data
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # start after warmup
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when market is choppy (range-bound)
        # Chop > 61.8 = ranging, Chop < 38.2 = trending
        if chop[i] <= 61.8:
            # In trending market, follow trend with KAMA
            if position == 0:  # look for entry
                if kama[i] > close[i] and trend_1d_aligned[i] > 0.5:  # KAMA below price = uptrend
                    position = 1
                    signals[i] = 0.25
                elif kama[i] < close[i] and trend_1d_aligned[i] <= 0.5:  # KAMA above price = downtrend
                    position = -1
                    signals[i] = -0.25
            elif position == 1:  # long
                # Exit: KAMA crosses above price (trend change) or RSI overbought
                if kama[i] < close[i] or rsi[i] > 70:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:  # short
                # Exit: KAMA crosses below price (trend change) or RSI oversold
                if kama[i] > close[i] or rsi[i] < 30:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:
            # In choppy market, mean reversion at RSI extremes
            if position == 0:  # look for entry
                if rsi[i] < 35 and close[i] > kama[i]:  # oversold but price above KAMA (bullish bias)
                    position = 1
                    signals[i] = 0.20
                elif rsi[i] > 65 and close[i] < kama[i]:  # overbought but price below KAMA (bearish bias)
                    position = -1
                    signals[i] = -0.20
            elif position == 1:  # long
                # Exit: RSI overbought or price crosses below KAMA
                if rsi[i] > 65 or close[i] < kama[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            elif position == -1:  # short
                # Exit: RSI oversold or price crosses above KAMA
                if rsi[i] < 35 or close[i] > kama[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals