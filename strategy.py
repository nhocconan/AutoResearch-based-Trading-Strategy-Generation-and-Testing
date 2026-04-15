#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel (20) for trend direction and 12h RSI(14) for mean reversion timing.
# In 1d uptrend (price > upper Donchian), wait for 12h RSI < 30 to go long (pullback entry).
# In 1d downtrend (price < lower Donchian), wait for 12h RSI > 70 to go short (bounce entry).
# Volume confirmation ensures momentum validity. Designed for low trade frequency (12-37/year) to minimize fee drag.
# Uses discrete position sizing (0.25) to reduce churn and manage drawdown in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d and 12h HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 30 or len(df_12h) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 12h Indicators: RSI(14) ===
    close_12h = df_12h['close'].values
    delta = pd.Series(close_12h).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only (reduces noise, keeps liquidity)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(rsi_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 1d uptrend (price > 1d upper Donchian)
        # 2. 12h RSI < 30 (oversold pullback)
        # 3. Volume confirmation
        if (close[i] > donchian_high_aligned[i]) and (rsi_12h_aligned[i] < 30) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 1d downtrend (price < 1d lower Donchian)
        # 2. 12h RSI > 70 (overbought bounce)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_aligned[i]) and (rsi_12h_aligned[i] > 70) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_RSI12h_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0