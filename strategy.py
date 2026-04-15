#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extremes (mean reversion) + 1w EMA200 trend filter + volume confirmation
# Williams %R < -80 = oversold (long), > -20 = overbought (short). Trade only when aligned with 1w EMA200 trend.
# Volume > 1.5x 20-bar SMA confirms momentum. Designed for low trade frequency (~30-60/year) to minimize fee drag.
# Works in both bull/bear: mean reversion in ranges, trend-filtered breaks in strong moves.

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
    
    # === 1d Indicators: Williams %R (14) ===
    highest_high_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_1d - df_1d['close'].values) / (highest_high_1d - lowest_low_1d)
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)  # avoid div by zero
    
    # Align to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # === 1w Indicators: EMA(200) for trend filter ===
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
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
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # Oversold (Williams %R < -80) AND price above 1w EMA200 (uptrend bias)
        if vol_confirm and williams_r_aligned[i] < -80 and close[i] > ema_200_1w_aligned[i]:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Overbought (Williams %R > -20) AND price below 1w EMA200 (downtrend bias)
        elif vol_confirm and williams_r_aligned[i] > -20 and close[i] < ema_200_1w_aligned[i]:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_WilliamsR_14_EMA200_1w_VolFilter_v1"
timeframe = "4h"
leverage = 1.0