#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: On daily timeframe, use KAMA to determine trend direction, RSI for momentum exhaustion, and Choppiness Index to filter ranging markets. 
# Long when KAMA is rising, RSI < 30 (oversold), and market is trending (CHOP < 38.2). 
# Short when KAMA is falling, RSI > 70 (overbought), and market is trending (CHOP < 38.2).
# Exit when RSI returns to neutral range (40-60) or market becomes choppy (CHOP > 61.8).
# Uses weekly timeframe for trend confirmation (price above/below weekly EMA200).
# Target: 15-25 trades/year with strict entry conditions to avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Indicators ===
    # KAMA ( Kaufman Adaptive Moving Average )
    def calculate_kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros_like(close)
        er[period:] = change[period-1:] / volatility[period-1:]
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[:] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    
    # RSI (14)
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
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    rsi_neutral = (rsi >= 40) & (rsi <= 60)
    
    # Choppiness Index (14)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period if i < period else np.mean(tr[i-period+1:i+1])
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                max_high[i] = np.max(high[:i+1])
                min_low[i] = np.min(low[:i+1])
            else:
                max_high[i] = np.max(high[i-period+1:i+1])
                min_low[i] = np.min(low[i-period+1:i+1])
        chop = np.where((max_high - min_low) != 0, 
                        100 * np.log10(np.sum(atr[i-period+1:i+1]) / (max_high - min_low)) / np.log10(period), 
                        50)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_trending = chop < 38.2
    chop_choppy = chop > 61.8
    
    # === Weekly Trend Filter (using mtf_data) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly EMA200 for trend direction
    close_1w = df_1w['close'].values
    ema_200_1w = np.zeros_like(close_1w)
    ema_200_1w[:] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_200_1w[i] = ema_200_1w[i-1] + 2/(200+1) * (close_1w[i] - ema_200_1w[i-1])
    
    # Align weekly EMA to daily timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    price_above_weekly_ema = close > ema_200_1w_aligned
    price_below_weekly_ema = close < ema_200_1w_aligned
    
    # === Signal Generation ===
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(100, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral or market becomes choppy
            if rsi_neutral[i] or chop_choppy[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral or market becomes choppy
            if rsi_neutral[i] or chop_choppy[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: KAMA rising, RSI oversold, trending market, price above weekly EMA
            if (kama_rising[i] and rsi_oversold[i] and chop_trending[i] and price_above_weekly_ema[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: KAMA falling, RSI overbought, trending market, price below weekly EMA
            elif (kama_falling[i] and rsi_overbought[i] and chop_trending[i] and price_below_weekly_ema[i]):
                position = -1
                signals[i] = -0.25
    
    return signals