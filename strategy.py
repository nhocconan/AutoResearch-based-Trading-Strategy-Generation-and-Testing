#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI mean-reversion and chop regime filter.
# Uses 1d KAMA (adaptive moving average) for trend direction and RSI(14) for entry timing.
# Long when price crosses above KAMA AND RSI < 40 (oversold) AND chop > 61.8 (ranging).
# Short when price crosses below KAMA AND RSI > 60 (overbought) AND chop > 61.8.
# Uses 1w ADX as regime filter: only trade when ADX < 25 (weak trend) to avoid whipsaw.
# Designed for 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by combining trend following with mean reversion in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on 1w
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        plus_di[period] = np.mean(plus_dm[1:period+1]) / atr[period] * 100 if atr[period] != 0 else 0
        minus_di[period] = np.mean(minus_dm[1:period+1]) / atr[period] * 100 if atr[period] != 0 else 0
        
        # Wilder's smoothing
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
            minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
        
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        dx = np.where((plus_di + minus_di) == 0, 0, dx)
        
        adx = np.zeros_like(high)
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # 1d KAMA (adaptive moving average)
    def calculate_kama(close, er_period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close.shape) > 1 else np.sum(np.abs(np.diff(close)))
        # For 1D array
        volatility = np.array([np.sum(np.abs(np.diff(close[i:i+er_period]))) if i+er_period <= len(close) else 0 
                              for i in range(len(close))])
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    
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
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Choppiness Index (14-period)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Sum of ATR over period
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Max and min range over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging market (chop > 61.8) and weak trend (ADX < 25)
        if chop[i] > 61.8 and adx_1w_aligned[i] < 25:
            # Long: price crosses above KAMA AND RSI < 40 (oversold)
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and rsi[i] < 40:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA AND RSI > 60 (overbought)
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and rsi[i] > 60:
                signals[i] = -0.25
                position = -1
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # In trending market or low chop, stay flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "1d_KAMA_RSI_Chop_ADXFilter"
timeframe = "1d"
leverage = 1.0