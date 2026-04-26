#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_And_Chop_Filter
Hypothesis: Trade daily KAMA trend direction with RSI momentum filter and choppiness regime avoidance.
KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI confirms momentum strength.
Choppiness Index filter ensures we only trade in trending regimes (CHOP < 38.2) or mean-revert in extreme chop (CHOP > 61.8).
Designed for low trade frequency (<15/year) to minimize fee drag and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Fix: calculate volatility correctly as sum of absolute changes over 10 periods
    volatility_series = pd.Series(np.abs(np.diff(close))).rolling(window=10, min_periods=10).sum().values
    volatility_series = np.concatenate([np.full(9, np.nan), volatility_series])  # align with change
    
    er = np.where(volatility_series > 0, change / volatility_series, 0)
    # Smoothing constants: fastest EMA(2), slowest EMA(30)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(13, np.nan), rsi])  # align
    
    # Calculate Choppiness Index (CHOP) over 14 periods
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    
    chop = np.where(
        (sum_atr > 0) & (range_hl > 0),
        100 * np.log10(sum_atr / range_hl) / np.log10(14),
        50
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), RSI (14), CHOP (14+14)
    start_idx = max(10, 14, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend direction: price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Momentum filter: RSI not extreme
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Regime filters
        trending_market = chop[i] < 38.2      # strong trend
        ranging_market = chop[i] > 61.8       # strong ranging (chop)
        
        if position == 0:
            # Long: price above KAMA + RSI not overbought + (trending OR ranging with mean reversion)
            long_signal = price_above_kama and rsi_not_overbought and (trending_market or ranging_market)
            
            # Short: price below KAMA + RSI not oversold + (trending OR ranging with mean reversion)
            short_signal = price_below_kama and rsi_not_oversold and (trending_market or ranging_market)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price crosses below KAMA OR RSI overbought OR strong trend breaks down
            if (close[i] < kama[i] or rsi[i] > 75 or chop[i] < 20):  # extreme low chop = exhaustion
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price crosses above KAMA OR RSI oversold OR strong trend breaks down
            if (close[i] > kama[i] or rsi[i] < 25 or chop[i] < 20):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_RSI_And_Chop_Filter"
timeframe = "1d"
leverage = 1.0