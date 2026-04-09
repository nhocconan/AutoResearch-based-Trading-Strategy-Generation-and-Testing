#!/usr/bin/env python3
# 1d_kama_rsi_chop_v2
# Hypothesis: 1d Kaufman Adaptive Moving Average (KAMA) trend with RSI mean reversion and choppiness filter.
# Uses 1d timeframe for low trade frequency (target: 20-80 over 4 years). KAMA adapts to market noise,
# providing reliable trend signals with reduced whipsaw. RSI(14) < 30 for long, > 70 for short entries
# only when price is near KAMA (mean reversion within trend). Choppiness Index (CHOP) > 50 ensures
# ranging/mean-reverting conditions, avoiding strong trends where mean reversion fails.
# Works in bull/bear markets: KAMA captures trend direction, RSI provides timely entries,
# chop filter avoids false signals in strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v2"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    if len(close) < er_period:
        return np.full_like(close, np.nan, dtype=float)
    
    close_series = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_series.diff(er_period))
    volatility = close_series.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constants
    sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift())
    tr3 = abs(low_series - close_series.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of True Range over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    hh = high_series.rolling(window=period, min_periods=period).max()
    ll = low_series.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    return chop.fillna(50).values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d HTF data ONCE before loop for KAMA, RSI, Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d indicators
    kama_1d = calculate_kama(close_1d, er_period=10, fast_period=2, slow_period=30)
    rsi_1d = calculate_rsi(close_1d, period=14)
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, period=14)
    
    # Align 1d indicators to 1d timeframe (completed daily candle only)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 60 or price > KAMA * 1.02 (take profit)
            if (rsi_1d_aligned[i] > 60) or (close[i] > kama_1d_aligned[i] * 1.02):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 40 or price < KAMA * 0.98 (take profit)
            if (rsi_1d_aligned[i] < 40) or (close[i] < kama_1d_aligned[i] * 0.98):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price < KAMA (mean reversion), RSI < 30, chop > 50 (ranging)
            if (close[i] < kama_1d_aligned[i]) and (rsi_1d_aligned[i] < 30) and (chop_1d_aligned[i] > 50):
                position = 1
                signals[i] = 0.25
            # Enter short: price > KAMA (mean reversion), RSI > 70, chop > 50 (ranging)
            elif (close[i] > kama_1d_aligned[i]) and (rsi_1d_aligned[i] > 70) and (chop_1d_aligned[i] > 50):
                position = -1
                signals[i] = -0.25
    
    return signals