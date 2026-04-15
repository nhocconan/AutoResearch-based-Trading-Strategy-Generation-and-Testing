#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakouts with 1w EMA200 trend filter and volume confirmation.
# In bull markets (price > 1w EMA200), long on upside Donchian breakout; in bear markets (price < 1w EMA200),
# short on downside Donchian breakout. Volume filter ensures momentum validity. Designed for low trade frequency
# (12-30/year) to minimize fee drag while adapting to bull/bear regimes via 1w EMA200.

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
    
    # Calculate Donchian bands
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(200) for bull/bear regime
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
            
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Bull market: price above 1w EMA200
        # Bear market: price below 1w EMA200
        
        is_bull = close[i] > ema_200_1w_aligned[i]
        is_bear = close[i] < ema_200_1w_aligned[i]
        
        # === LONG CONDITIONS ===
        # Only in bull market: long on upside Donchian breakout with volume
        if is_bull and vol_confirm:
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Only in bear market: short on downside Donchian breakout with volume
        elif is_bear and vol_confirm:
            if close[i] < lower_20_aligned[i]:
                signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_1wEMA200_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0