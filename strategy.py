#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day KAMA (Kaufman Adaptive Moving Average) with RSI momentum and Chop filter.
# KAMA adapts to market noise - fast in trends, slow in ranging markets.
# RSI (14) provides momentum confirmation - only trade when momentum aligns with trend.
# Chop index filters for trending markets only (Chop < 38.2) to avoid whipsaws in ranges.
# Daily timeframe with weekly trend filter for higher reliability.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio (ER) for KAMA
    def er(close, period=10):
        change = np.abs(np.diff(close, period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        # Vectorized calculation
        result = np.full(len(close), np.nan)
        for i in range(period, len(close)):
            if np.sum(np.abs(np.diff(close[i-period:i+1]))) > 0:
                result[i] = np.abs(close[i] - close[i-period]) / np.sum(np.abs(np.diff(close[i-period:i+1])))
            else:
                result[i] = 0
        return result
    
    er_values = er(close_1d, 10)
    # Smoothing constants
    sc = (er_values * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.full(len(close_1d), np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI on daily
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(close), np.nan)
        avg_loss = np.full(len(close), np.nan)
        
        # First average
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            # Wilder smoothing
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values
    
    rsi_values = rsi(close_1d, 14)
    
    # Calculate Chop index on daily
    def chop(high, low, close, period=14):
        atr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr[i] = tr
        
        # Smoothed ATR
        atr_sum = np.zeros(len(close))
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Max/min range
        max_high = np.zeros(len(close))
        min_low = np.zeros(len(close))
        for i in range(period-1, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        chop_values = np.zeros(len(close))
        for i in range(period-1, len(close)):
            if atr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
                chop_values[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop_values[i] = 50
        return chop_values
    
    chop_values = chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Weekly trend filter: price above/below weekly KAMA
    close_1w = df_1w['close'].values
    er_1w = er(close_1w, 10)
    sc_1w = (er_1w * (2/2 - 2/30) + 2/30) ** 2
    kama_1w = np.full(len(close_1w), np.nan)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        if not np.isnan(sc_1w[i]):
            kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
        else:
            kama_1w[i] = kama_1w[i-1]
    
    # Align all data to daily timeframe (our trading timeframe)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(kama_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Conditions:
        # 1. Price relative to KAMA (trend direction)
        # 2. RSI momentum (not overbought/oversold extremes)
        # 3. Chop filter (trending market only)
        # 4. Weekly trend alignment
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        weekly_kama = kama_1w_aligned[i]
        
        # Only trade in trending markets (Chop < 38.2)
        trending_market = chop_val < 38.2
        
        # Weekly trend filter
        weekly_uptrend = price > weekly_kama
        weekly_downtrend = price < weekly_kama
        
        if position == 0:
            # Long: price above KAMA, RSI not overbought, weekly uptrend, trending market
            if (price > kama_val and 
                rsi_val < 70 and 
                weekly_uptrend and 
                trending_market):
                position = 1
                signals[i] = position_size
            # Short: price below KAMA, RSI not oversold, weekly downtrend, trending market
            elif (price < kama_val and 
                  rsi_val > 30 and 
                  weekly_downtrend and 
                  trending_market):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below KAMA OR RSI overbought OR weekly trend changes OR market becomes ranging
            if (price < kama_val or 
                rsi_val > 75 or 
                not weekly_uptrend or 
                not trending_market):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price above KAMA OR RSI oversold OR weekly trend changes OR market becomes ranging
            if (price > kama_val or 
                rsi_val < 25 or 
                not weekly_downtrend or 
                not trending_market):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0