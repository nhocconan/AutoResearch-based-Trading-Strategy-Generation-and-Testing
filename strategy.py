#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend filter + volume confirmation
# - Primary signal: Donchian channel breakout on 1d timeframe - long when price breaks above upper band, short when breaks below lower band
# - Trend filter: 1w HMA(21) - only take longs when price above HMA, shorts when price below HMA (higher timeframe alignment)
# - Volume confirmation: 1d volume > 20-period median volume (avoid low-participation breakouts)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, HMA filter avoids counter-trend trades, volume confirmation ensures validity

name = "1d_1w_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Pre-compute 1w HMA(21) for trend filter
    close_1w = df_1w['close'].values
    # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full(len(values), np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = np.array([wma(close_1w[i:i+half_len], half_len) if i+half_len <= len(close_1w) else np.nan 
                         for i in range(len(close_1w))])
    wma_full = np.array([wma(close_1w[i:i+21], 21) if i+21 <= len(close_1w) else np.nan 
                         for i in range(len(close_1w))])
    raw_hma = 2 * wma_half - wma_full
    hma_21_1w = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(raw_hma) else np.nan 
                          for i in range(len(raw_hma))])
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Pre-compute Donchian(20) channels on 1d timeframe
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1d volume > 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(hma_21_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower band OR price crosses below 1w HMA21
            if close_1d[i] < lowest_low[i] or close_1d[i] < hma_21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper band OR price crosses above 1w HMA21
            if close_1d[i] > highest_high[i] or close_1d[i] > hma_21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and 1w HMA21 filter
            # Long: price breaks above Donchian upper band AND volume regime AND price above 1w HMA21
            if (close_1d[i] > highest_high[i] and 
                volume_regime[i] and 
                close_1d[i] > hma_21_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower band AND volume regime AND price below 1w HMA21
            elif (close_1d[i] < lowest_low[i] and 
                  volume_regime[i] and 
                  close_1d[i] < hma_21_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals