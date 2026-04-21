#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v2
Hypothesis: Daily KAMA trend with RSI momentum and choppiness regime filter. 
Long when KAMA rising + RSI > 50 + choppy market (mean reversion favorable). 
Short when KAMA falling + RSI < 50 + choppy market. 
Uses 1d primary timeframe with 1w HTF for trend confirmation to reduce noise and avoid false signals in low-volatility regimes.
Designed for low trade frequency (~10-25/year) to minimize fee drag and improve generalization across bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1w trend filter: 21-period EMA ===
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === 1d KAMA (adaptive moving average) ===
    close = prices['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # 10-period sum of absolute changes
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = change / volatility
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1d RSI (14-period) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    # === 1d Choppiness Index (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    # True range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    # Sum of true range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Avoid division by zero and invalid values
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # warmup for indicators
        # Skip if indicators not ready
        if (np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        kama_now = kama[i]
        kama_prev = kama[i-1]
        rsi_now = rsi[i]
        chop_now = chop[i]
        weekly_trend = ema_21_1w_aligned[i]
        
        # Determine KAMA direction
        kama_rising = kama_now > kama_prev
        kama_falling = kama_now < kama_prev
        
        if position == 0:
            # Long: KAMA rising + RSI > 50 + choppy market (CHOP > 50) + price above weekly EMA
            if kama_rising and rsi_now > 50 and chop_now > 50 and price_close > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + RSI < 50 + choppy market (CHOP > 50) + price below weekly EMA
            elif kama_falling and rsi_now < 50 and chop_now > 50 and price_close < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or chop becomes too low (trending market)
            if position == 1:
                # Exit long: KAMA falling OR RSI < 50 OR chop <= 50 (trending) OR price below weekly EMA
                if (not kama_rising) or rsi_now < 50 or chop_now <= 50 or price_close < weekly_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: KAMA rising OR RSI > 50 OR chop <= 50 (trending) OR price above weekly EMA
                if (not kama_falling) or rsi_now > 50 or chop_now <= 50 or price_close > weekly_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0