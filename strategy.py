#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with volume confirmation
# and 1d EMA50 trend filter. In trending markets (price outside 4h Donchian), 
# trade breakout continuation on 1h with volume spike. In ranging markets (price 
# inside 4h Donchian), no trade. Designed for low trade frequency (15-35/year) 
# to minimize fee drag while capturing strong moves. Uses 4h for signal direction, 
# 1h only for entry timing with volume filter to avoid false breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper/lower bands (20-period)
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(50) for trend bias
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === ENTRY LOGIC ===
        # Long: price breaks above 4h Donchian high + volume confirm + above 1d EMA50
        # Short: price breaks below 4h Donchian low + volume confirm + below 1d EMA50
        if vol_confirm:
            if close[i] > donchian_high_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.20
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.20
            else:
                signals[i] = 0.0  # flat
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Donchian20_VolumeFilter_EMA50_v1"
timeframe = "1h"
leverage = 1.0