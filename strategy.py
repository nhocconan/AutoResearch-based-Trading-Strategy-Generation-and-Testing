#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakout with volume confirmation and 1w EMA trend filter.
# Uses 1d Donchian(20) for breakout signals, filtered by 1w EMA50 for trend bias and volume spike for momentum confirmation.
# Session filter (08-20 UTC) reduces noise trades. Designed for low trade frequency (20-40/year) to minimize fee drag.
# Works in bull/bear: 1w EMA50 avoids counter-trend trades, Donchian breakouts capture sustained momentum with volume confirmation.

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
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Donchian upper = max(high, lookback=20)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower = min(low, lookback=20)
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(50) for trend bias
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Donchian upper (20-period high)
        # 2. 1w price above EMA50 (bullish trend bias)
        # 3. Volume confirmation
        if (close[i] > donch_high_aligned[i] and
            close[i] > ema_50_1w_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Donchian lower (20-period low)
        # 2. 1w price below EMA50 (bearish trend bias)
        # 3. Volume confirmation
        elif (close[i] < donch_low_aligned[i] and
              close[i] < ema_50_1w_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_1d_Donchian20_1w_EMA50_VolFilter_v1"
timeframe = "4h"
leverage = 1.0