#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams %R (14) for mean reversion and 1d EMA(200) for trend filter.
# Long when: 12h Williams %R < -80 (oversold) AND price > 1d EMA(200) (uptrend)
# Short when: 12h Williams %R > -20 (overbought) AND price < 1d EMA(200) (downtrend)
# Volume confirmation ensures momentum validity. Designed for low trade frequency (12-25/year) to minimize fee drag.
# Williams %R identifies exhaustion points in trends, while 1d EMA(200) ensures we trade with the higher timeframe trend.
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought bounces in downtrend).

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
    
    # Get 12h and 1d HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Williams %R (14) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low + 1e-10)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # === 1d Indicators: EMA(200) ===
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. 12h Williams %R < -80 (oversold)
        # 2. Price > 1d EMA(200) (uptrend)
        # 3. Volume confirmation
        if (williams_r_aligned[i] < -80) and (close[i] > ema_200_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. 12h Williams %R > -20 (overbought)
        # 2. Price < 1d EMA(200) (downtrend)
        # 3. Volume confirmation
        elif (williams_r_aligned[i] > -20) and (close[i] < ema_200_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR14_EMA200_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0