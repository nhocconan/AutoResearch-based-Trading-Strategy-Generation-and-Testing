# 1d_KAMA_RSI_Chop_Filter_v1
# Hypothesis: Daily KAMA trend filter + RSI mean reversion + Choppiness regime filter. 
# In trending markets (CHOP < 38.2): follow KAMA direction. In ranging markets (CHOP > 61.8): fade RSI extremes.
# Works in both bull and bear by adapting to market regime. Target: 10-20 trades/year.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily KAMA (trend)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # needs correction
    # Recalculate volatility properly
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Smooth volatility for ER calculation
    volatility_smooth = pd.Series(volatility).rolling(window=10, min_periods=1).mean().values
    er = np.zeros(n)
    er[10:] = change[10:] / (volatility_smooth[10:] + 1e-10)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    # ATR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Max high - min low over period
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    # Chop
    chop = np.zeros(n)
    chop[13:] = 100 * np.log10(atr_sum[13:] / (range_max_min[13:] + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(kama[i]) or \
           np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Regime-based entry
            if chop[i] < 38.2:  # Trending regime
                # Follow KAMA direction with weekly filter
                if price > kama[i] and price > ema34_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif price < kama[i] and price < ema34_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif chop[i] > 61.8:  # Ranging regime
                # Fade RSI extremes
                if rsi[i] < 30:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70:  # Overbought
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long
            if chop[i] < 38.2:  # Trending: exit when price < KAMA
                if price < kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging: exit when RSI > 50 or chop leaves range
                if rsi[i] > 50 or chop[i] < 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit short
            if chop[i] < 38.2:  # Trending: exit when price > KAMA
                if price > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging: exit when RSI < 50 or chop leaves range
                if rsi[i] < 50 or chop[i] < 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals