#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop Filter
# Uses Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI for momentum,
# and Choppiness Index to filter range-bound markets.
# KAMA adapts to market noise, reducing whipsaws in choppy conditions.
# Works in bull/bear by following adaptive trend with momentum confirmation.
# Target: 30-100 total trades over 4 years (7-25/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    # Proper volatility calculation for ER
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Use rolling sum of absolute changes
    volatility = pd.Series(close).diff().abs().rolling(window=10, min_periods=1).sum().values
    
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc[0] = 0
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_max_min = highest_high - lowest_low
    range_max_min = np.where(range_max_min == 0, 1e-10, range_max_min)
    
    chop = 100 * np.log10(atr * 14 / range_max_min) / np.log10(14)
    
    # Weekly trend filter (1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_kama = np.zeros_like(weekly_close)
    weekly_kama[0] = weekly_close[0]
    # Recalculate weekly KAMA (same parameters)
    weekly_change = np.abs(weekly_close - np.roll(weekly_close, 10))
    weekly_change[0:10] = 0
    weekly_volatility = pd.Series(weekly_close).diff().abs().rolling(window=10, min_periods=1).sum().values
    weekly_er = np.where(weekly_volatility != 0, weekly_change / weekly_volatility, 0)
    weekly_sc = (weekly_er * (fast_sc - slow_sc) + slow_sc) ** 2
    weekly_sc[0] = 0
    for i in range(1, len(weekly_close)):
        weekly_kama[i] = weekly_kama[i-1] + weekly_sc[i] * (weekly_close[i] - weekly_kama[i-1])
    weekly_kama_aligned = align_htf_to_ltf(prices, df_1w, weekly_kama)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for RSI and other indicators
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(weekly_kama_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Chop filter: only trade when market is not too choppy (CHOP < 61.8)
        if chop[i] > 61.8:
            # In choppy/ranging market, stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, and weekly KAMA trending up
            if price > kama[i] and rsi[i] > 50 and weekly_kama_aligned[i] > weekly_kama_aligned[i-1]:
                position = 1
                signals[i] = position_size
            # Short: price below KAMA, RSI < 50, and weekly KAMA trending down
            elif price < kama[i] and rsi[i] < 50 and weekly_kama_aligned[i] < weekly_kama_aligned[i-1]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI < 40
            if price < kama[i] or rsi[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI > 60
            if price > kama[i] or rsi[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0