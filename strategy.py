#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with 4h EMA(34) trend filter and volume confirmation.
# In 1d uptrend (price > 1d upper Donchian) and 4h EMA rising, go long on pullback to 4h EMA.
# In 1d downtrend (price < 1d lower Donchian) and 4h EMA falling, go short on bounce to 4h EMA.
# Volume confirmation ensures momentum validity. Designed for low trade frequency (20-40/year) to minimize fee drag.
# Works in bull via breakouts and in bear via mean reversion to EMA within HTF trend.

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
    
    # Get 1d and 4h HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    if len(df_1d) < 30 or len(df_4h) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # === 4h Indicators: EMA(34) ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
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
        if (np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or
            np.isnan(ema_34_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 1d uptrend (price > 1d upper Donchian)
        # 2. 4h EMA rising (EMA > EMA_prev)
        # 3. Price near 4h EMA (pullback entry)
        # 4. Volume confirmation
        if (close[i] > donchian_high_1d_aligned[i]) and \
           (ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1]) and \
           (abs(close[i] - ema_34_4h_aligned[i]) / ema_34_4h_aligned[i] < 0.02) and \
           vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 1d downtrend (price < 1d lower Donchian)
        # 2. 4h EMA falling (EMA < EMA_prev)
        # 3. Price near 4h EMA (bounce entry)
        # 4. Volume confirmation
        elif (close[i] < donchian_low_1d_aligned[i]) and \
             (ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1]) and \
             (abs(close[i] - ema_34_4h_aligned[i]) / ema_34_4h_aligned[i] < 0.02) and \
             vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_1dDonchian20_4hEMA34_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0