#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly KAMA for trend direction
    close_1w = df_1w['close'].values
    # Calculate Efficiency Ratio (ER)
    change_1w = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility_1w = np.abs(np.diff(close_1w))
    er_1w = change_1w / np.maximum(volatility_1w, 1e-10)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc_1w = (er_1w * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama_1w = np.zeros_like(close_1w)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    # Shift to use only completed weekly bars
    kama_1w = np.roll(kama_1w, 1)
    kama_1w[0] = np.nan
    # Align weekly KAMA to daily
    kama_1d = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Choppiness Index for regime filter
    atr1 = high[1:] - low[1:]
    atr2 = np.abs(high[1:] - close[:-1])
    atr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(atr1, np.maximum(atr2, atr3))])
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_1d[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below weekly KAMA
        price_above_kama = close[i] > kama_1d[i]
        price_below_kama = close[i] < kama_1d[i]
        
        # RSI conditions
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        # Chop regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending
        chop_ranging = chop[i] > 61.8
        chop_trending = chop[i] < 38.2
        
        # Trading logic
        if chop_ranging:
            # In ranging market: mean reversion at RSI extremes
            if rsi_oversold and price_above_kama:
                # Oversold but above weekly trend -> long
                if position != 1:
                    position = 1
                    signals[i] = 0.25
            elif rsi_overbought and price_below_kama:
                # Overbought but below weekly trend -> short
                if position != -1:
                    position = -1
                    signals[i] = -0.25
            # Exit when RSI returns to neutral
            elif position == 1 and rsi[i] >= 50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and rsi[i] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # In trending market: follow trend with KAMA
            if price_above_kama and rsi[i] > 50:
                # Uptrend with bullish momentum
                if position != 1:
                    position = 1
                    signals[i] = 0.25
            elif price_below_kama and rsi[i] < 50:
                # Downtrend with bearish momentum
                if position != -1:
                    position = -1
                    signals[i] = -0.25
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily KAMA + RSI + Chop regime strategy.
# Uses weekly KAMA for trend direction to avoid false signals in chop.
# In ranging markets (CHOP > 61.8): mean reversion at RSI extremes (30/70) with trend filter.
# In trending markets (CHOP < 38.2): follow weekly KAMA trend with RSI confirmation.
# Chop filter adapts to market conditions, reducing whipsaw in both bull and bear markets.
# Position size: 0.25 to limit drawdown. Target: 30-100 trades over 4 years.
# Weekly timeframe reduces noise compared to daily-only indicators.
# Works in bull markets by following trend, in bear markets by mean-reverting in ranges.