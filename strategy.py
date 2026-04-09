#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# - Primary signal: 1d price breaks above/below Donchian(20) channel
# - Trend filter: 1w EMA50 - price must be above EMA for longs, below for shorts
# - Volume confirmation: 1d volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, 1w EMA50 filter ensures alignment with higher timeframe trend,
#   volume confirmation reduces false signals, ATR-based stoploss manages risk

name = "1d_1w_donchian_ema_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA50 for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe (completed 1w bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower OR price crosses below 1w EMA50
            if close[i] < lowest_low_20[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper OR price crosses above 1w EMA50
            if close[i] > highest_high_20[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and 1w EMA50 filter
            # Long: price breaks above Donchian upper AND volume regime AND price above 1w EMA50
            if close[i] > highest_high_20[i] and volume_regime[i] and close[i] > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower AND volume regime AND price below 1w EMA50
            elif close[i] < lowest_low_20[i] and volume_regime[i] and close[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals