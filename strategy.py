#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation
# Long when price breaks above upper Donchian channel AND price > 1d EMA50 (uptrend)
# Short when price breaks below lower Donchian channel AND price < 1d EMA50 (downtrend)
# Volume confirmation ensures breakout validity. Designed for low trade frequency (20-40/year)
# to minimize fee drag while capturing trending moves in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(50) for trend bias
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Calculate rolling high/low for Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period volume SMA
    vol_series = pd.Series(volume)
    vol_sma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_sma_20 * 1.5)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # Price breaks above upper Donchian channel AND in uptrend (price > 1d EMA50) AND volume confirmation
        if (close[i] > donchian_high[i] and 
            close[i] > ema_50_1d_aligned[i] and 
            vol_confirm[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below lower Donchian channel AND in downtrend (price < 1d EMA50) AND volume confirmation
        elif (close[i] < donchian_low[i] and 
              close[i] < ema_50_1d_aligned[i] and 
              vol_confirm[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_EMA50_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0