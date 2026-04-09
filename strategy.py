#!/usr/bin/env python3
# 1d_kama_rsi_chop_regime_v1
# Hypothesis: 1d strategy using KAMA (Kaufman Adaptive Moving Average) for trend direction, RSI for momentum confirmation, and Choppiness Index regime filter.
# Long when KAMA is rising, RSI > 50, and chop > 61.8 (ranging market) for mean reversion longs.
# Short when KAMA is falling, RSI < 50, and chop > 61.8 (ranging market) for mean reversion shorts.
# Uses weekly trend filter (1w KAMA direction) to avoid counter-trend trades.
# Exit when RSI crosses back to neutral (50) or chop < 38.2 (strong trend) to avoid whipsaws.
# Designed for low trade frequency (7-25/year) to minimize fee drag on 1d timeframe.
# Works in both bull and bear markets: KAMA adapts to changing volatility, RSI captures momentum extremes, chop filter identifies mean-reversion regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation (Kaufman Adaptive Moving Average)
    close_s = pd.Series(close)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(close_s.diff(periods=10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    fastest = 2 / (2 + 1)  # EMA(2)
    slowest = 2 / (30 + 1) # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close, dtype=float)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Choppiness Index (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.log10(14) / (highest_high - lowest_low))
    
    # Weekly HTF trend filter (1w KAMA direction)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    # Calculate 1w KAMA (same parameters)
    change_1w = np.abs(close_1w_s.diff(periods=10))
    volatility_1w = close_1w_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er_1w = change_1w / volatility_1w.replace(0, np.nan)
    sc_1w = (er_1w * (fastest - slowest) + slowest) ** 2
    kama_1w = np.zeros_like(close_1w, dtype=float)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        if np.isnan(sc_1w[i]):
            kama_1w[i] = kama_1w[i-1]
        else:
            kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    # Align 1w KAMA to 1d timeframe (completed weekly bars only)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    # Weekly trend: rising if current > previous, falling if current < previous
    kama_1w_rising = np.zeros(n, dtype=bool)
    kama_1w_falling = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(kama_1w_aligned[i]) and i > 0:
            kama_1w_rising[i] = kama_1w_aligned[i] > kama_1w_aligned[i-1]
            kama_1w_falling[i] = kama_1w_aligned[i] < kama_1w_aligned[i-1]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_values[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(kama_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop[i] > 61.8
        # Strong trend filter: chop < 38.2 indicates strong trend (avoid mean reversion)
        strong_trend = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50 OR strong trend develops
            if rsi_values[i] < 50 and rsi_values[i-1] >= 50:
                position = 0
                signals[i] = 0.0
            elif strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50 OR strong trend develops
            if rsi_values[i] > 50 and rsi_values[i-1] <= 50:
                position = 0
                signals[i] = 0.0
            elif strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for entry conditions with weekly trend filter
            kama_rising = kama[i] > kama[i-1]
            kama_falling = kama[i] < kama[i-1]
            rsi_above_50 = rsi_values[i] > 50
            rsi_below_50 = rsi_values[i] < 50
            
            # Long: KAMA rising, RSI > 50, ranging market, and weekly KAMA rising
            if (kama_rising and rsi_above_50 and ranging_market and 
                kama_1w_rising[i]):
                position = 1
                signals[i] = 0.25
            # Short: KAMA falling, RSI < 50, ranging market, and weekly KAMA falling
            elif (kama_falling and rsi_below_50 and ranging_market and 
                  kama_1w_falling[i]):
                position = -1
                signals[i] = -0.25
    
    return signals