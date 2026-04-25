#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: Daily KAMA trend direction combined with RSI extremes and Choppiness Index regime filter.
KAMA adapts to market noise, reducing false signals in choppy markets. RSI < 30 for long, > 70 for short in trending regimes (ADX > 25).
Targets 7-25 trades/year by requiring: 1) KAMA trend alignment, 2) RSI extreme, 3) ADX > 25 (trending market).
Designed to work in both bull and bear markets via trend-following with momentum confirmation and regime filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1w data for ADX regime filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['close']).shift()).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close']).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Directional Movement
    dm_plus = pd.Series(df_1w['high']).diff()
    dm_minus = -pd.Series(df_1w['low']).diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    # Smoothed values
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    dpi_100 = 100 * (dm_plus.ewm(alpha=1/14, adjust=False).mean() / atr)
    dmi_100 = 100 * (dm_minus.ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = (abs(dpi_100 - dmi_100) / (dpi_100 + dmi_100)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    
    # 1d data for KAMA (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, 10))
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=0)[:len(change)]
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1d data for RSI (loaded ONCE)
    rsi_period = 14
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    avg_gain[rsi_period] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period] = np.mean(loss[:rsi_period])
    for i in range(rsi_period+1, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i-1]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i-1]) / rsi_period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Look for entry signals with regime filter
            # Long: price > KAMA, RSI < 30 (oversold), trending market
            long_signal = (curr_close > kama_aligned[i]) and (rsi_aligned[i] < 30) and trending
            # Short: price < KAMA, RSI > 70 (overbought), trending market
            short_signal = (curr_close < kama_aligned[i]) and (rsi_aligned[i] > 70) and trending
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price < KAMA (trend change) or RSI > 50 (momentum fade)
            if (curr_close < kama_aligned[i]) or (rsi_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price > KAMA (trend change) or RSI < 50 (momentum fade)
            if (curr_close > kama_aligned[i]) or (rsi_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0