#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop Regime Filter
# Uses KAMA to identify trend direction on 1d, confirmed by RSI(14) momentum and filtered by
# Choppiness Index (14) > 61.8 for range regime. Works in both bull and bear markets by
# combining trend following with mean reversion in range regimes.
# Target: 30-100 total trades over 4 years (7-25/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1w data (higher timeframe for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === KAMA (Kaufman Adaptive Moving Average) on 1d ===
    def kama(close, er_len=10, fast=2, slow=30):
        if len(close) < er_len:
            return np.full_like(close, np.nan, dtype=float)
        change = np.abs(np.diff(close, n=er_len))
        abs_change = np.abs(np.diff(close, n=1))
        er = np.zeros_like(close)
        er[er_len:] = change[er_len:] / np.where(np.sum(abs_change.reshape(-1, er_len), axis=1) == 0, 1, np.sum(abs_change.reshape(-1, er_len), axis=1))
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_arr = np.zeros_like(close)
        kama_arr[:] = close[0]
        for i in range(1, len(close)):
            kama_arr[i] = kama_arr[i-1] + sc[i] * (close[i] - kama_arr[i-1])
        return kama_arr
    
    kama_1d = kama(close_1d)
    kama_1d_prev = np.roll(kama_1d, 1)
    kama_1d_prev[0] = kama_1d[0]
    
    # === RSI(14) on 1d ===
    def rsi(close, period=14):
        if len(close) < period + 1:
            return np.full_like(close, 50.0, dtype=float)
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
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        rsi_arr = 100 - (100 / (1 + rs))
        return rsi_arr
    
    rsi_1d = rsi(close_1d)
    
    # === Choppiness Index (14) on 1d ===
    def choppiness_index(high, low, close, period=14):
        if len(close) < period:
            return np.full_like(close, 50.0, dtype=float)
        atr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
        atr[0] = high[0] - low[0]
        sum_atr = np.zeros_like(close)
        for i in range(period, len(close)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(period-1, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if hh[i] - ll[i] == 0:
                chop[i] = 50.0
            else:
                chop[i] = 100 * np.log10(sum_atr[i] / (hh[i] - ll[i])) / np.log10(period)
        return chop
    
    chop_1d = choppiness_index(high_1d, low_1d, close_1d)
    
    # === 1w EMA200 for trend filter ===
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama_1d[i]) or 
            np.isnan(kama_1d_prev[i]) or
            np.isnan(rsi_1d[i]) or
            np.isnan(chop_1d[i]) or
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        kama_val = kama_1d[i]
        kama_prev = kama_1d_prev[i]
        rsi_val = rsi_1d[i]
        chop_val = chop_1d[i]
        ema200 = ema200_1w_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
            atr_1d[0] = high_1d[0] - low_1d[0]
            atr_ma = np.zeros_like(atr_1d)
            for j in range(14, len(atr_1d)):
                atr_ma[j] = np.mean(atr_1d[j-13:j+1])
            atr_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
            atr_1d[0] = high_1d[0] - low_1d[0]
            atr_ma = np.zeros_like(atr_1d)
            for j in range(14, len(atr_1d)):
                atr_ma[j] = np.mean(atr_1d[j-13:j+1])
            atr_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when KAMA turns down or RSI overbought
            if kama_val < kama_prev or rsi_val > 70:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when KAMA turns up or RSI oversold
            if kama_val > kama_prev or rsi_val < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade in range regime (Choppiness > 61.8)
            if chop_val > 61.8:
                # Go long when KAMA turns up and RSI > 50
                if kama_val > kama_prev and rsi_val > 50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when KAMA turns down and RSI < 50
                elif kama_val < kama_prev and rsi_val < 50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0