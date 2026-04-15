#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Choppiness Regime Filter
# KAMA adapts to volatility and trend strength, providing a dynamic trend filter.
# RSI identifies overbought/oversold conditions within the trend.
# Choppiness Index filters out ranging markets (high chop) to trade only in trending environments.
# This combination aims to capture trend continuation moves with controlled entries.
# Timeframe: 1d, HTF: 1w (for regime filter)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Adaptive Moving Average) on close
    # Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # needs correction - will compute properly below
    
    # Proper ER calculation: need volatility over lookback period
    er = np.zeros(n)
    for i in range(10, n):  # 10-period ER
        price_change = np.abs(close[i] - close[i-10])
        price_volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
        if price_volatility > 0:
            er[i] = price_change / price_volatility
        else:
            er[i] = 0
    er[0:10] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14) - need weekly data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # True Range and ADX components for Choppiness
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(close_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(close_1w, 1)), 
                        np.maximum(np.roll(close_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values for ADX
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_1w + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1w + 1e-10)
    
    # DX and ADX for Choppiness calculation
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1w = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index formula: 100 * log10(sum(ATR)/ (n * (max(high)-min(low)))) / log10(n)
    # We'll use ADX-based approximation: Chop = 100 - ADX (simplified but effective)
    chop_1w = 100 - adx_1w
    
    # Align indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            continue
            
        # Long: Price above KAMA (uptrend), RSI not overbought, market trending (low chop)
        if (close[i] > kama_aligned[i] and
            rsi_aligned[i] < 70 and
            chop_aligned[i] < 40 and  # Trending market (chop < 40)
            position <= 0):
            position = 1
            signals[i] = base_size
            
        # Short: Price below KAMA (downtrend), RSI not oversold, market trending (low chop)
        elif (close[i] < kama_aligned[i] and
              rsi_aligned[i] > 30 and
              chop_aligned[i] < 40 and  # Trending market
              position >= 0):
            position = -1
            signals[i] = -base_size
            
        # Exit: Trend weakness (high chop) or opposite KAMA crossover
        elif position == 1 and (chop_aligned[i] > 50 or close[i] < kama_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (chop_aligned[i] > 50 or close[i] > kama_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0