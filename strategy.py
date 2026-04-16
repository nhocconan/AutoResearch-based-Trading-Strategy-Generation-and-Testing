#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND 1d choppiness < 61.8 (trending market).
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND 1d choppiness < 61.8.
# Uses discrete position size 0.25. Donchian breakouts capture strong moves, volume confirmation ensures participation,
# choppiness filter avoids whipsaws in ranging markets. Designed to work in both bull (breakout long) and bear (breakdown short).
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    
    # Get 1d data once before loop for volume and choppiness filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume MA and choppiness calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma)
    
    # === 1d Indicators: Choppiness Index (14-period) ===
    # True Range
    tr1 = pd.Series(high_1d).rolling(window=2).max().values - pd.Series(low_1d).rolling(window=2).min().values
    tr2 = np.abs(pd.Series(high_1d).rolling(window=2).shift(1).values - pd.Series(close_1d).rolling(window=2).shift(1).values)
    tr3 = np.abs(pd.Series(low_1d).rolling(window=2).shift(1).values - pd.Series(close_1d).rolling(window=2).shift(1).values)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(np.sum(tr) / (np.log10(14) * (hh - ll))) if (hh - ll) != 0 else 50
    # Fix: calculate properly per bar
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr / (np.log10(14) * (hh - ll)))
    chop = np.where((hh - ll) == 0, 50, chop)  # avoid division by zero
    
    # Trending market: chop < 61.8
    trending_market = chop < 61.8
    
    # Align 1d indicators to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    trending_market_aligned = align_htf_to_ltf(prices, df_1d, trending_market)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian, volume MA, choppiness)
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(trending_market_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        dch_high = donchian_high[i]
        dch_low = donchian_low[i]
        vol_spike = volume_spike_aligned[i]
        trending = trending_market_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Donchian low or conditions deteriorate
            if price < dch_low or not vol_spike or not trending:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Donchian high or conditions deteriorate
            if price > dch_high or not vol_spike or not trending:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike AND trending market
            if price > dch_high and vol_spike and trending:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian low AND volume spike AND trending market
            elif price < dch_low and vol_spike and trending:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0