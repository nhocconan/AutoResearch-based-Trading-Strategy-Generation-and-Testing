# 4H_ELIXIR: KAMA + RSI + CHOPPINESS + VOLUME
# Hypothesis: In both bull and bear markets, KAMA captures trend direction with minimal lag.
# RSI filters overbought/oversold conditions. Choppiness regime avoids whipsaws in sideways markets.
# Volume confirms institutional participation. Designed for fewer, higher-quality trades.

#!/usr/bin/env python3
name = "4H_ELIXIR_KAMA_RSI_CHOP_VOL"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- CHOPPINESS INDEX (14) - Regime Filter ---
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 / (highest_high - lowest_low)) / np.log10(14)
    
    # --- KAMA (10, 2, 30) - Adaptive Trend ---
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # Manual sum of abs changes
    # Fix volatility calculation: rolling sum of absolute changes
    volatility_series = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=10, min_periods=10).sum().values
    volatility_series[0] = 0  # First value undefined
    er = np.where(volatility_series != 0, change / volatility_series, 0)
    sc = (er * (0.0645 - 0.0625) + 0.0625) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- RSI (14) - Momentum Filter ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # --- DAILY VOLUME FILTER ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.5 * vol_ma20_1d_aligned
    
    # --- SESSION FILTER: 08-20 UTC ---
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending (trend follow)
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        if position == 0:
            # Long: Price > KAMA + RSI < 70 (not overbought) + volume + trending OR (ranging and RSI < 40)
            if close[i] > kama[i] and volume_filter[i]:
                if (is_trending and rsi[i] < 70) or (is_ranging and rsi[i] < 40):
                    signals[i] = 0.25
                    position = 1
            # Short: Price < KAMA + RSI > 30 (not oversold) + volume + trending OR (ranging and RSI > 60)
            elif close[i] < kama[i] and volume_filter[i]:
                if (is_trending and rsi[i] > 30) or (is_ranging and rsi[i] > 60):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Price < KAMA OR RSI > 80 (extreme overbought)
            if close[i] < kama[i] or rsi[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price > KAMA OR RSI < 20 (extreme oversold)
            if close[i] > kama[i] or rsi[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals