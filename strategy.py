#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel (20) for long-term trend and 1d RSI(2) for short-term mean reversion.
# In 1w uptrend (price > weekly upper Donchian), wait for 1d RSI(2) < 10 to go long (extreme oversold).
# In 1w downtrend (price < weekly lower Donchian), wait for 1d RSI(2) > 90 to go short (extreme overbought).
# Volume confirmation (current volume > 1.5x 20-day average) ensures momentum validity.
# Designed for very low trade frequency (~10-20/year) to minimize fee drag while capturing major reversals.
# Works in both bull (trend-following pullbacks) and bear (counter-trend bounces) markets.

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
    
    # Get 1w and 1d HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1w Indicators: Donchian Channel (20) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # === 1d Indicators: RSI(2) ===
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_2_1d = 100 - (100 / (1 + rs))
    rsi_2_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_2_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-day volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or
            np.isnan(rsi_2_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 1w uptrend (price > weekly upper Donchian)
        # 2. 1d RSI(2) < 10 (extremely oversold)
        # 3. Volume confirmation
        if (close[i] > donchian_high_1w_aligned[i]) and (rsi_2_1d_aligned[i] < 10) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 1w downtrend (price < weekly lower Donchian)
        # 2. 1d RSI(2) > 90 (extremely overbought)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_1w_aligned[i]) and (rsi_2_1d_aligned[i] > 90) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_1w_RSI2_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0