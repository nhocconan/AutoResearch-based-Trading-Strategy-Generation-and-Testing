#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
# - Primary signal: Donchian breakout - long when price > upper band, short when price < lower band
# - Trend filter: 1d EMA50 - price must be above EMA for longs, below for shorts
# - Volume confirmation: 4h volume > 20-period median volume (avoid low-participation signals)
# - Exit: price retracement to 20-period EMA or opposite Donchian band touch
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian captures breakouts, EMA50 filter ensures alignment with higher timeframe trend

name = "4h_1d_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute indicators on primary timeframe (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period EMA for exit
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 20 EMA OR touches lower Donchian band
            if close[i] < ema_20[i] or close[i] <= donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 20 EMA OR touches upper Donchian band
            if close[i] > ema_20[i] or close[i] >= donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and 1d EMA50 filter
            # Long: price breaks above upper Donchian band AND volume regime AND price above 1d EMA50
            if (close[i] > donchian_upper[i] and 
                volume_regime[i] and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower Donchian band AND volume regime AND price below 1d EMA50
            elif (close[i] < donchian_lower[i] and 
                  volume_regime[i] and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals