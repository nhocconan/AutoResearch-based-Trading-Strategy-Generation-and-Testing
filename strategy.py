#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA direction + RSI(14) + choppiness regime filter on 1w
    # Long when: KAMA trending up (close > KAMA) AND RSI < 70 AND chop < 61.8 (trending regime)
    # Short when: KAMA trending down (close < KAMA) AND RSI > 30 AND chop < 61.8 (trending regime)
    # Exit when: chop > 61.8 (ranging regime) OR RSI extremes (RSI>80 for long, RSI<20 for short)
    # Uses discrete sizing (0.25) targeting 30-100 trades over 4 years.
    # KAMA adapts to market noise, RSI avoids overextended entries, chop filter ensures we only trade in trending markets.
    # Works in bull (trend continuation) and bear (avoids whipsaws in ranging markets).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for choppiness regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate KAMA(10) on 1d close
    def kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Smoothing Constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA
        kama_values = np.full_like(close, np.nan, dtype=float)
        kama_values[period] = close[period]
        for i in range(period+1, len(close)):
            kama_values[i] = kama_values[i-1] + sc[i] * (close[i] - kama_values[i-1])
        return kama_values
    
    kama_values = kama(close, 10, 2, 30)
    
    # Calculate RSI(14) on 1d close
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan, dtype=float)
        avg_loss = np.full_like(close, np.nan, dtype=float)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values
    
    rsi_values = rsi(close, 14)
    
    # Calculate Choppiness Index(14) on 1w
    def choppy_index(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of TR over period
        tr_sum = np.full_like(close, np.nan, dtype=float)
        for i in range(period-1, len(tr)):
            tr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.full_like(high, np.nan, dtype=float)
        min_low = np.full_like(low, np.nan, dtype=float)
        for i in range(period-1, len(high)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        # Choppiness Index
        chop = np.full_like(close, np.nan, dtype=float)
        for i in range(period-1, len(close)):
            if tr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
                chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop_values = choppy_index(high_1w, low_1w, close_1w, 14)
    
    # Align HTF indicators to 1d timeframe (wait for completed 1w bar)
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama_values)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi_values)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: close > KAMA = up trend, close < KAMA = down trend
        kama_up = close[i] > kama_aligned[i]
        kama_down = close[i] < kama_aligned[i]
        
        # RSI conditions: avoid overextended entries
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        
        # Chop regime: only trade when trending (chop < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        # Exit conditions: chop > 61.8 (ranging) or RSI extremes
        exit_long = chop_aligned[i] > 61.8 or rsi_aligned[i] > 80
        exit_short = chop_aligned[i] > 61.8 or rsi_aligned[i] < 20
        
        # Entry conditions
        long_entry = kama_up and rsi_not_overbought and trending_regime and position != 1
        short_entry = kama_down and rsi_not_oversold and trending_regime and position != -1
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_kama_rsi_chop_regime_v1"
timeframe = "1d"
leverage = 1.0