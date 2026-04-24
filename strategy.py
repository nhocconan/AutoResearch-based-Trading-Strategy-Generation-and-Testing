#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend + RSI mean reversion + chop regime filter.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for regime filter (choppiness index > 61.8 = range, < 38.2 = trending).
- Entry: Long when KAMA bullish (price > KAMA) AND RSI < 40 AND chop regime = trending.
         Short when KAMA bearish (price < KAMA) AND RSI > 60 AND chop regime = trending.
- Exit: Opposite KAMA cross OR RSI crosses 50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- KAMA adapts to market noise, reducing false signals in choppy markets.
- RSI provides mean-reversion entries within the trend.
- Chop regime filter ensures we only trade in trending markets, avoiding whipsaws.
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
- Estimated trades: ~60 total over 4 years (~15/year) based on KAMA/RSI/chop alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close = pd.Series(close)
    change = abs(close - close.shift(er_period))
    volatility = abs(close.diff()).rolling(window=er_period, min_periods=er_period).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = [close.iloc[0]]  # seed
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close.iloc[i] - kama[-1]))
    return np.array(kama)

def rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close = pd.Series(close)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def true_range(high, low, close_prev):
    """Calculate True Range."""
    tr1 = high - low
    tr2 = abs(high - close_prev)
    tr3 = abs(low - close_prev)
    return np.maximum(tr1, np.maximum(tr2, tr3))

def atr(high, low, close, period=14):
    """Calculate Average True Range."""
    tr = true_range(high, low, np.roll(close, 1))
    tr[0] = np.nan  # first TR is undefined
    atr_vals = pd.Series(tr).rolling(window=period, min_periods=period).mean()
    return atr_vals.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d indicators
    kama_vals = kama(close, er_period=10, fast=2, slow=30)
    rsi_vals = rsi(close, period=14)
    atr_vals = atr(high, low, close, period=14)
    
    # Calculate 1w chop regime (Choppiness Index)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Choppiness Index on 1w: high/low range vs ATR sum
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    atr_1w = atr(high_1w, low_1w, close_1w, period=14)
    atr_sum = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_1w = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w, additional_delay_bars=0)
    
    # Regime: chop > 61.8 = range (choppy), chop < 38.2 = trending
    chop_trending = chop_1w_aligned < 38.2
    chop_ranging = chop_1w_aligned > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for KAMA/RSI/ATR/chop
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(atr_vals[i]) or
            np.isnan(chop_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite KAMA cross OR RSI crosses 50 in opposite direction
        if position != 0:
            # Exit long: price < KAMA OR RSI > 50
            if position == 1:
                if curr_close < kama_vals[i] or rsi_vals[i] > 50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > KAMA OR RSI < 50
            elif position == -1:
                if curr_close > kama_vals[i] or rsi_vals[i] < 50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: KAMA aligned + RSI extreme + trending chop regime
        if position == 0:
            # Long: KAMA bullish (price > KAMA) AND RSI < 40 AND chop regime = trending
            if curr_close > kama_vals[i] and rsi_vals[i] < 40 and chop_trending[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA bearish (price < KAMA) AND RSI > 60 AND chop regime = trending
            elif curr_close < kama_vals[i] and rsi_vals[i] > 60 and chop_trending[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopRegime_v1"
timeframe = "1d"
leverage = 1.0