#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + Choppiness regime filter
# Long when KAMA bullish (close > KAMA) + RSI < 40 (oversold) + CHOP > 61.8 (range)
# Short when KAMA bearish (close < KAMA) + RSI > 60 (overbought) + CHOP > 61.8 (range)
# Uses weekly trend filter: only trade in direction of weekly KAMA
# Target: 50-100 total trades over 4 years with mean reversion in ranging markets
# ATR-based stoploss to limit drawdown

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d KAMA calculation (ER=10, FC=2, SC=30)
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close))
        er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1d = kama(close)
    
    # 1d RSI(14)
    def rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = rsi(close)
    
    # 1d Choppiness Index (CHOP)
    def chop(high, low, close, period=14):
        atr = np.maximum(np.abs(high - low), 
                        np.maximum(np.abs(high - np.roll(close, 1)), 
                                 np.abs(np.roll(close, 1) - low)))
        atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        return chop
    
    chop_1d = chop(high, low, close)
    
    # Weekly KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    kama_1w = kama(close_1w)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]) or 
            np.isnan(chop_1d[i]) or np.isnan(kama_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation
            atr_approx = np.max([high[i] - low[i], 
                               np.abs(high[i] - close[i-1]), 
                               np.abs(low[i] - close[i-1])])
            if close[i] < entry_price - 2.0 * atr_approx:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI overbought or trend change
            elif rsi_1d[i] > 70 or close[i] < kama_1d[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            atr_approx = np.max([high[i] - low[i], 
                               np.abs(high[i] - close[i-1]), 
                               np.abs(low[i] - close[i-1])])
            if close[i] > entry_price + 2.0 * atr_approx:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI oversold or trend change
            elif rsi_1d[i] < 30 or close[i] > kama_1d[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with weekly trend filter
            # Long: KAMA bullish + RSI oversold + choppy market + weekly uptrend
            if (close[i] > kama_1d[i] and 
                rsi_1d[i] < 40 and
                chop_1d[i] > 61.8 and
                close[i] > kama_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: KAMA bearish + RSI overbought + choppy market + weekly downtrend
            elif (close[i] < kama_1d[i] and 
                  rsi_1d[i] > 60 and
                  chop_1d[i] > 61.8 and
                  close[i] < kama_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals