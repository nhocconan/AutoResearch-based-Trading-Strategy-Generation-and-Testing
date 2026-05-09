#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d HTF for trend filtering. Uses 1d Donchian breakout (20-period) 
# with volume confirmation and volatility filter to avoid chop. Designed for fewer trades (target 50-150/4y) 
# to reduce fee drag. Works in both bull/bear via trend filter and volatility regime filter.
name = "12h_Donchian20_Breakout_1dTrend_Volume_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volatility filter: ATR(14) ratio to avoid choppy markets
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Avoid division by zero
    atr_ratio = atr / (close + 1e-10)
    atr_ma = pd.Series(atr_ratio).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5 x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(atr_ma[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extremely high or low volatility regimes
        vol_filter = (atr_ratio[i] > atr_ma[i] * 0.5) & (atr_ratio[i] < atr_ma[i] * 2.0)
        
        # Volume condition: current volume > 1.5 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: Break above Donchian high with uptrend, volume spike, and vol filter
            if close[i] > donchian_high_12h[i] and close[i] > ema50_12h[i] and vol_spike and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with downtrend, volume spike, and vol filter
            elif close[i] < donchian_low_12h[i] and close[i] < ema50_12h[i] and vol_spike and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below Donchian low OR trend turns down
            if close[i] < donchian_low_12h[i] or close[i] < ema50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above Donchian high OR trend turns up
            if close[i] > donchian_high_12h[i] or close[i] > ema50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals