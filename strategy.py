#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend filter + volume confirmation
# - Primary signal: 4h close breaks above/below Donchian(20) channel from prior 20 bars
# - Trend filter: 12h HMA(21) - price must be above HMA for longs, below for shorts
# - Volume confirmation: 4h volume > 1.5 * 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian provides adaptive structure, HMA filter ensures alignment with
#   higher timeframe trend, reducing false signals in choppy/range markets

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h HMA(21) for trend direction
    close_12h = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_12h).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21_12h = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    
    # Align 12h HMA21 to 4h timeframe (completed 12h bar only)
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian(20) channel - use prior 20 bars only (no look-ahead)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 4h volume regime: volume > 1.5 * 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(hma_21_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR price crosses below HMA21
            if close[i] < donchian_low[i] or close[i] < hma_21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR price crosses above HMA21
            if close[i] > donchian_high[i] or close[i] > hma_21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and HMA21 filter
            # Long: close breaks above Donchian high AND volume regime AND price above HMA21
            if close[i] > donchian_high[i] and volume_regime[i] and close[i] > hma_21_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: close breaks below Donchian low AND volume regime AND price below HMA21
            elif close[i] < donchian_low[i] and volume_regime[i] and close[i] < hma_21_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals