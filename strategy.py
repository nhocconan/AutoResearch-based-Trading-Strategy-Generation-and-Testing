#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h HMA trend + volume confirmation
# - Primary signal: Donchian(20) breakout on 4h - long when price breaks above upper band, short when breaks below lower band
# - Trend filter: 12h HMA(21) - price must be above HMA for longs, below for shorts (aligns with higher timeframe trend)
# - Volume confirmation: 4h volume > 1.5 * 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian captures breakouts in trends, HMA filter avoids counter-trend trades, volume confirms conviction

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h HMA(21) for trend filter
    close_12h = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_12h).rolling(window=half_len, min_periods=half_len).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True).values
    wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True).values
    hma_21_12h = pd.Series(2 * wma_half - wma_full).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True).values
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Pre-compute 4h Donchian(20) channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h volume regime: volume > 1.5 * 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(hma_21_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower band OR price crosses below 12h HMA
            if close_4h[i] < donchian_lower[i] or close_4h[i] < hma_21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper band OR price crosses above 12h HMA
            if close_4h[i] > donchian_upper[i] or close_4h[i] > hma_21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and 12h HMA filter
            # Long: price breaks above Donchian upper band AND volume regime AND price above 12h HMA
            if (close_4h[i] > donchian_upper[i] and 
                volume_regime[i] and 
                close_4h[i] > hma_21_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower band AND volume regime AND price below 12h HMA
            elif (close_4h[i] < donchian_lower[i] and 
                  volume_regime[i] and 
                  close_4h[i] < hma_21_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals