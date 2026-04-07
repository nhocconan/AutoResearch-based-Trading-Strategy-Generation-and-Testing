#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily KAMA with RSI and Choppiness Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend signals.
# In trending markets (low Choppiness): KAMA direction signals trend strength.
# In ranging markets (high Choppiness): RSI extremes signal mean reversion.
# Volume confirms institutional participation. Daily trend filter ensures alignment with higher timeframe.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_daily_kama_rsi_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA, RSI, Choppiness, and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily KAMA (ER=10, FAST=2, SLOW=30)
    close_daily = df_daily['close'].values
    change = np.abs(np.diff(close_daily, prepend=close_daily[0]))
    volatility = np.abs(np.diff(close_daily))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_daily)
    kama[0] = close_daily[0]
    for i in range(1, len(close_daily)):
        kama[i] = kama[i-1] + sc[i] * (close_daily[i] - kama[i-1])
    
    # Calculate daily RSI (14-period)
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily Choppiness Index (14-period)
    atr = np.zeros_like(close_daily)
    tr1 = np.abs(np.diff(df_daily['high'].values))
    tr2 = np.abs(np.diff(df_daily['low'].values))
    tr3 = np.abs(df_daily['high'].values[:-1] - df_daily['low'].values[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.insert(tr, 0, tr[0])  # same length as close
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    high_low = df_daily['high'].values - df_daily['low'].values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh_ll = pd.Series(df_daily['high'].values).rolling(window=14, min_periods=14).max().values - \
            pd.Series(df_daily['low'].values).rolling(window=14, min_periods=14).min().values
    chop = np.where(hh_ll != 0, 100 * np.log10(atr_sum / hh_ll) / np.log10(14), 50)
    
    # Daily trend filter: price vs 50 EMA
    close_series = pd.Series(close_daily)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align daily data to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_daily, chop)
    ema_50_aligned = align_htf_to_ltf(prices, df_daily, ema_50)
    
    # Volume filter: volume > 1.5x 50-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA turns down or RSI overbought or chop high (range) or trend fails
            if (kama_aligned[i] < kama_aligned[i-1] or rsi_aligned[i] > 70 or
                chop_aligned[i] > 61.8 or close[i] < ema_50_aligned[i] or
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: KAMA turns up or RSI oversold or chop high (range) or trend fails
            if (kama_aligned[i] > kama_aligned[i-1] or rsi_aligned[i] < 30 or
                chop_aligned[i] > 61.8 or close[i] > ema_50_aligned[i] or
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: KAMA rising + RSI oversold + low chop (trend) + volume + above EMA50
            if (kama_aligned[i] > kama_aligned[i-1] and rsi_aligned[i] < 30 and
                chop_aligned[i] < 61.8 and close[i] > ema_50_aligned[i] and
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: KAMA falling + RSI overbought + low chop (trend) + volume + below EMA50
            elif (kama_aligned[i] < kama_aligned[i-1] and rsi_aligned[i] > 70 and
                  chop_aligned[i] < 61.8 and close[i] < ema_50_aligned[i] and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals