#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R for mean reversion and 1w EMA200 for trend filter.
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short) in ranging markets.
# 1w EMA200 acts as trend filter: only long when price > EMA200, short when price < EMA200.
# Volume confirmation ensures momentum validity. Designed for low trade frequency (15-25/year)
# to minimize fee drag while capturing mean reversion in BTC/ETH ranging markets.

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
    
    # === 1d Indicators: Williams %R(14) ===
    highest_high_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max()
    lowest_low_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min()
    williams_r_1d = -100 * ((highest_high_1d - df_1d['close'].values) / (highest_high_1d - lowest_low_1d))
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d.values)
    
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
        if (np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # Williams %R oversold (< -80) AND price above 1w EMA200 (uptrend bias) AND volume confirmation
        if (williams_r_1d_aligned[i] < -80 and 
            close[i] > ema_200_1w_aligned[i] and 
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Williams %R overbought (> -20) AND price below 1w EMA200 (downtrend bias) AND volume confirmation
        elif (williams_r_1d_aligned[i] > -20 and 
              close[i] < ema_200_1w_aligned[i] and 
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR_1d_EMA200_VolFilter_v1"
timeframe = "6h"
leverage = 1.0