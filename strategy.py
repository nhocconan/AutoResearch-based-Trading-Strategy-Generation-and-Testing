#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI + chop filter (weekly)
# Long when KAMA rising + RSI < 45 + weekly chop < 61.8 (trending)
# Short when KAMA falling + RSI > 55 + weekly chop < 61.8
# Uses 1d price for KAMA/RSI, weekly for chop regime filter
# Target: 50-100 total trades over 4 years with controlled risk
# ATR-based stoploss to limit drawdown

name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation (10-period)
    def calculate_kama(close, period=10):
        change = np.abs(np.diff(close, n=period))
        volatility = np.abs(np.diff(close)).rolling(window=period, min_periods=1).sum()
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/2 - 2/30) + 2/30) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[period-1] = close[period-1]
        for i in range(period, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10)
    
    # RSI (14-period)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(close, np.nan, dtype=float)
        avg_loss = np.full_like(close, np.nan, dtype=float)
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # 1w data for chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Chop calculation (14-period)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = np.where(np.isnan(tr), 0, tr)
        for i in range(2, len(atr)):
            if not np.isnan(tr[i]):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        highest_high[0] = high[0]
        lowest_low[0] = low[0]
        for i in range(1, len(close)):
            highest_high[i] = max(highest_high[i-1], high[i])
            lowest_low[i] = min(lowest_low[i-1], low[i])
        
        hhll_diff = highest_high - lowest_low
        atr_sum = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                atr_sum[i] = np.sum(atr[max(0, i-period+1):i+1])
            else:
                atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        chop = np.where(hhll_diff != 0, 100 * np.log10(atr_sum / hhll_diff) / np.log10(period), 50)
        return chop
    
    chop_1w = calculate_chop(high_1w, low_1w, close_1w, 14)
    
    # Align 1w chop to 1d timeframe
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # ATR for stoploss (14-period)
    def calculate_atr(high, low, close, period=14):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros_like(close)
        atr[1:] = np.where(np.isnan(tr), 0, tr)
        for i in range(2, len(atr)):
            if not np.isnan(tr[i]):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA falling or RSI > 55 or chop > 61.8 (ranging)
            elif (kama[i] < kama[i-1] or rsi[i] > 55 or chop_1w_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA rising or RSI < 45 or chop > 61.8 (ranging)
            elif (kama[i] > kama[i-1] or rsi[i] < 45 or chop_1w_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: KAMA direction + RSI extreme + trending chop
            # Long: KAMA rising + RSI < 45 + chop < 61.8 (trending)
            if (kama[i] > kama[i-1] and 
                rsi[i] < 45 and
                chop_1w_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: KAMA falling + RSI > 55 + chop < 61.8 (trending)
            elif (kama[i] < kama[i-1] and 
                  rsi[i] > 55 and
                  chop_1w_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals