#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_And_Chop_Filter_V1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, 
combined with RSI(14) for momentum confirmation and Choppiness Index(14) for regime filtering.
Long when: price > KAMA(10) AND RSI > 50 AND CHOP > 61.8 (ranging market mean reversion to upside)
Short when: price < KAMA(10) AND RSI < 50 AND CHOP > 61.8 (ranging market mean reversion to downside)
Exit when opposite KAMA crossover occurs or RSI reaches extreme levels (70/30) for profit taking.
This strategy targets low-frequency trades (7-25/year) by using daily timeframe with multiple confirmation filters,
works in both bull and bear markets by adapting to ranging regimes where mean reversion is strongest.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')  # Weekly trend filter
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === Daily Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA(10, 2, 30) - Adaptive Trend ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    # Handle array dimensions correctly
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(1, np.nan), volatility])
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    # === RSI(14) ===
    delta = np.diff(close)
    delta = np.concatenate([np.array([np.nan]), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index(14) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) > 0, chop, 50)
    
    # === Weekly Trend Filter (EMA34 on 1w) ===
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        weekly_trend_up = ema_1w_aligned[i] < price  # Price above weekly EMA = bullish bias
        weekly_trend_down = ema_1w_aligned[i] > price  # Price below weekly EMA = bearish bias
        
        if position == 0:
            # Long: price above KAMA (bullish trend) + RSI > 50 (bullish momentum) + Chop > 61.8 (rangy market for mean reversion)
            if (price > kama[i] and rsi[i] > 50 and chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (bearish trend) + RSI < 50 (bearish momentum) + Chop > 61.8 (rangy market for mean reversion)
            elif (price < kama[i] and rsi[i] < 50 and chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA (trend change) OR RSI > 70 (overbought) OR Chop < 38.2 (trending market - breakout)
            if (price < kama[i] or rsi[i] > 70 or chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA (trend change) OR RSI < 30 (oversold) OR Chop < 38.2 (trending market - breakdown)
            if (price > kama[i] or rsi[i] < 30 or chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_RSI_And_Chop_Filter_V1"
timeframe = "1d"
leverage = 1.0