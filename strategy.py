#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA200 trend + volume confirmation
# - Primary signal: Donchian(20) breakout on 1d timeframe - long when price breaks above upper band, short when breaks below lower band
# - Trend filter: 1w EMA200 - price must be above EMA for longs, below for shorts (higher timeframe alignment)
# - Volume confirmation: 1d volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, EMA200 filter avoids counter-trend trades, volume confirms validity

name = "1d_1w_donchian_ema200_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Pre-compute 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute Donchian(20) on 1d timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # 1d volume regime: volume > 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian middle OR price crosses below 1w EMA200
            if close[i] < donchian_middle[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian middle OR price crosses above 1w EMA200
            if close[i] > donchian_middle[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and 1w EMA200 filter
            # Long: price breaks above Donchian upper AND volume regime AND price above 1w EMA200
            if (close[i] > donchian_upper[i] and 
                volume_regime[i] and 
                close[i] > ema_200_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower AND volume regime AND price below 1w EMA200
            elif (close[i] < donchian_lower[i] and 
                  volume_regime[i] and 
                  close[i] < ema_200_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals