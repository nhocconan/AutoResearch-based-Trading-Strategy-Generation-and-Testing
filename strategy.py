#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakouts with volume confirmation and ATR-based stoploss.
# In trending markets, breakouts above/below 1d Donchian(20) capture momentum; in ranging markets, 
# price reverts to the 1d EMA50. Volume filter ensures breakout validity. Designed for low trade 
# frequency (20-50/year) to minimize fee drag while adapting to trend via higher timeframe structure.
# Works in both bull and bear markets by using Donchian breakouts (trend following) and EMA reversion (mean reversion).

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (20) and EMA(50) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian Channel: 20-period high/low
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # EMA(50) for trend bias and mean reversion target
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema = ema_50_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. Donchian breakout above upper band (trend following)
        # 2. OR price below EMA and reverting upward (mean reversion in uptrug)
        if vol_confirm:
            if (price > upper) or \
               (price < ema * 0.98 and price > lower):  # slight dip below EMA but above lower band
                signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Donchian breakdown below lower band (trend following)
        # 2. OR price above EMA and reverting downward (mean reversion in downtrend)
        elif vol_confirm:
            if (price < lower) or \
               (price > ema * 1.02 and price < upper):  # slight rally above EMA but below upper band
                signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_1d_Donchian20_EMA50_VolFilter_v1"
timeframe = "4h"
leverage = 1.0