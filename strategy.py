#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_ChopFilter_v1
Hypothesis: Daily KAMA trend direction combined with RSI momentum and Choppiness index regime filter.
KAMA adapts to market noise, reducing whipsaws in chop while capturing trends. RSI confirms momentum.
Chop filter avoids trend-following in ranging markets (chop > 61.8) and avoids mean-reversion in strong trends (chop < 38.2).
Works in bull/bear via adaptive trend filter and regime-aware positioning. Targets 1d timeframe for low trade frequency.
"""

name = "1d_1w_KAMA_RSI_ChopFilter_v1"
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
    
    # KAMA: adaptive moving average
    def calculate_kama(close_series, er_period=10, fast=2, slow=30):
        change = np.abs(np.diff(close_series, n=1))
        volatility = np.abs(np.diff(close_series, n=10))  # approximate
        er = np.zeros_like(close_series)
        er[10:] = change[9:] / (volatility + 1e-10)
        er = np.where(volatility == 0, 0, er)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close_series)
        kama[0] = close_series[0]
        for i in range(1, len(close_series)):
            kama[i] = kama[i-1] + sc[i] * (close_series[i] - kama[i-1])
        return kama
    
    # Choppiness Index
    def calculate_chop(high_series, low_series, close_series, period=14):
        atr = np.zeros_like(close_series)
        tr = np.zeros_like(close_series)
        for i in range(1, len(close_series)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        for i in range(period, len(close_series)):
            atr[i] = np.mean(tr[i-period+1:i+1])
        
        sum_atr = np.zeros_like(close_series)
        for i in range(period, len(close_series)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        
        hh = np.zeros_like(close_series)
        ll = np.zeros_like(close_series)
        for i in range(period, len(close_series)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        chop = np.zeros_like(close_series)
        for i in range(period, len(close_series)):
            if hh[i] - ll[i] > 0:
                chop[i] = 100 * np.log10(sum_atr[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    # RSI
    def calculate_rsi(close_series, period=14):
        delta = np.diff(close_series)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close_series)
        avg_loss = np.zeros_like(close_series)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close_series)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate indicators
    kama = calculate_kama(close, er_period=10, fast=2, slow=30)
    chop = calculate_chop(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(30, n):
        if (np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA up, chop < 61.8 (not too choppy), RSI > 50, price > 1w EMA
            if (kama[i] > kama[i-1] and 
                chop[i] < 61.8 and 
                rsi[i] > 50 and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down, chop < 61.8, RSI < 50, price < 1w EMA
            elif (kama[i] < kama[i-1] and 
                  chop[i] < 61.8 and 
                  rsi[i] < 50 and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA down OR chop > 61.8 (choppy) OR RSI < 40
            if (kama[i] < kama[i-1] or 
                chop[i] > 61.8 or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA up OR chop > 61.8 OR RSI > 60
            if (kama[i] > kama[i-1] or 
                chop[i] > 61.8 or 
                rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals