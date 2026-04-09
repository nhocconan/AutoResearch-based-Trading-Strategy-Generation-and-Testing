#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d/1w regime filter
# - Entry: Price breaks above/below Donchian(20) channel + volume > 1.5x 20-bar average
# - Direction filter: Only take longs when price > 1d EMA(50) and 1w close > 1w open (bullish weekly)
#                 Only take shorts when price < 1d EMA(50) and 1w close < 1w open (bearish weekly)
# - Exit: Price returns to Donchian midpoint OR 1d EMA(50) crossover
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 20-40 trades/year (80-160 total over 4 years) to avoid fee drag
# - Combines structure (Donchian) with volume and multi-timeframe alignment for robustness

name = "4h_1d_1w_donchian_breakout_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    weekly_bullish = close_1w > open_1w  # True for bullish weekly candle
    
    # Align 1d EMA and 1w direction to 4h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # 4h Donchian(20) channel
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2
    
    # Volume confirmation: current volume > 1.5x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: price returns to midpoint or weekly turns bearish
            if close[i] <= donchian_mid[i] or weekly_bullish_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price returns to midpoint or weekly turns bullish
            if close[i] >= donchian_mid[i] or weekly_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and regime alignment
            bullish_breakout = (close[i] > highest_20[i] and 
                               volume_confirm[i] and 
                               close[i] > ema_50_1d_aligned[i] and 
                               weekly_bullish_aligned[i] > 0.5)
            
            bearish_breakout = (close[i] < lowest_20[i] and 
                               volume_confirm[i] and 
                               close[i] < ema_50_1d_aligned[i] and 
                               weekly_bullish_aligned[i] < 0.5)
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals