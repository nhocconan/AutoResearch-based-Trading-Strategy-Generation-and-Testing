#!/usr/bin/env python3
# 1d_weekly_kama_rsi_regime_v1
# Hypothesis: On 1d timeframe, use weekly HTF regime filter (EMA50) with KAMA trend and RSI mean reversion.
# Long: price > KAMA(10) AND RSI(14) < 40 AND weekly close > weekly EMA50 (bullish regime)
# Short: price < KAMA(10) AND RSI(14) > 60 AND weekly close < weekly EMA50 (bearish regime)
# Exit: opposite signal or RSI crosses 50 (mean reversion)
# Uses discrete sizing 0.25 to limit fee drag. Target 20-60 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_kama_rsi_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA(10, 2, 30)
    # ER = |Close - Close[10]| / Sum(|Close - Close[1]|, 10)
    change = np.abs(np.diff(close, n=10))  # |Close - Close[10]|
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    volatility = np.where(np.arange(n) >= 10, volatility[9:] - np.concatenate([np.zeros(9), volatility[:-9]]), volatility)
    er = np.zeros(n)
    er[10:] = change[9:] / np.where(volatility[9:] > 0, volatility[9:], 1)
    # Smooth constant
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full(n, np.nan)
    kama[9] = np.mean(close[:10])
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for EMA regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on weekly
    ema_50_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(df_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 / (50 + 1)) + (ema_50_1w[i-1] * (49 / (50 + 1)))
    
    # Align weekly EMA to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        weekly_ema = ema_50_1w_aligned[i]
        
        if np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(weekly_ema):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        regime_bullish = close[i] > weekly_ema
        regime_bearish = close[i] < weekly_ema
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 OR short signal
            if rsi_val >= 50 or (price < kama_val and rsi_val > 60 and regime_bearish):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 OR long signal
            if rsi_val <= 50 or (price > kama_val and rsi_val < 40 and regime_bullish):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price > KAMA AND RSI < 40 AND bullish weekly regime
            if price > kama_val and rsi_val < 40 and regime_bullish:
                position = 1
                signals[i] = 0.25
            # Short entry: price < KAMA AND RSI > 60 AND bearish weekly regime
            elif price < kama_val and rsi_val > 60 and regime_bearish:
                position = -1
                signals[i] = -0.25
    
    return signals