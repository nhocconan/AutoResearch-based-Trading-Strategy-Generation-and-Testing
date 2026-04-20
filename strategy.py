#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d trend filter (price > SMA200)
# Works in bull: breakouts capture momentum; works in bear: SMA200 filter avoids shorts in strong downtrends, longs only in residual strength
# Target: 20-40 trades/year (80-160 total over 4 years) to avoid fee drag
name = "4h_Donchian20_Volume_SMA200Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for SMA200 filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === Daily SMA200 for trend filter ===
    close_1d = df_1d['close'].values
    sma200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    # === 4h Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    
    # Upper band: highest high of past 20 bars (excluding current)
    high_series = pd.Series(high)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    # Lower band: lowest low of past 20 bars (excluding current)
    low_series = pd.Series(low)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # === Volume confirmation: volume > 1.5x 20-period average ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        vol_ratio_val = vol_ratio[i]
        sma200_val = sma200_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(vol_ratio_val) or np.isnan(sma200_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian band with volume confirmation and price > daily SMA200
            if close_val > upper_val and vol_ratio_val > 1.5 and close_val > sma200_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian band with volume confirmation and price < daily SMA200
            elif close_val < lower_val and vol_ratio_val > 1.5 and close_val < sma200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below lower Donchian band OR volume drops
            if close_val < lower_val or vol_ratio_val < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above upper Donchian band OR volume drops
            if close_val > upper_val or vol_ratio_val < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals