# 1d_1w_kama_rsi_chop_filter_v3
# Hypothesis: Daily KAMA trend with RSI momentum and Chop index regime filter to avoid false signals in ranging markets.
# Uses 1w trend filter to ensure alignment with higher timeframe trend.
# Designed for low trade frequency (<25/year) to minimize fee drag and work in both bull and bear markets.

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly KAMA for trend direction
    def calculate_kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1w = calculate_kama(close_1w)
    
    # Daily indicators for entry
    # Daily KAMA for entry signal
    kama_1d = calculate_kama(close)
    
    # Daily RSI for momentum
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
    
    rsi_1d = calculate_rsi(close)
    
    # Chop index for regime detection
    def calculate_chop(high, low, close, period=14):
        tr1 = np.abs(high - low)
        tr2 = np.abs(np.roll(high, 1) - close)
        tr3 = np.abs(np.roll(low, 1) - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros_like(close)
        for i in range(period, len(close)):
            atr[i] = np.mean(tr[i-period+1:i+1])
        max_h = np.zeros_like(close)
        min_l = np.zeros_like(close)
        for i in range(period, len(close)):
            max_h[i] = np.max(high[i-period+1:i+1])
            min_l[i] = np.min(low[i-period+1:i+1])
        chop = np.where((max_h - min_l) != 0, 
                       100 * np.log10(atr.sum() / (max_h - min_l)) / np.log10(period), 
                       50)
        return chop
    
    chop_1d = calculate_chop(high, low, close)
    
    # Align weekly KAMA to daily timeframe
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(kama_1d[i]) or 
            np.isnan(rsi_1d[i]) or np.isnan(chop_1d[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > kama_1w_aligned[i]
        weekly_downtrend = close[i] < kama_1w_aligned[i]
        
        # Chop filter: avoid ranging markets (Chop > 61.8) and extreme trends (Chop < 38.2)
        # We want moderate trending conditions: Chop between 38.2 and 61.8
        chop_condition = (chop_1d[i] >= 38.2) and (chop_1d[i] <= 61.8)
        
        # Entry conditions
        if weekly_uptrend and chop_condition and position != 1:
            # Long: price above daily KAMA and RSI > 50 (momentum confirmation)
            if close[i] > kama_1d[i] and rsi_1d[i] > 50:
                position = 1
                signals[i] = 0.25
        elif weekly_downtrend and chop_condition and position != -1:
            # Short: price below daily KAMA and RSI < 50 (momentum confirmation)
            if close[i] < kama_1d[i] and rsi_1d[i] < 50:
                position = -1
                signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (not weekly_uptrend or not chop_condition):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not weekly_downtrend or not chop_condition):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_kama_rsi_chop_filter_v3"
timeframe = "1d"
leverage = 1.0