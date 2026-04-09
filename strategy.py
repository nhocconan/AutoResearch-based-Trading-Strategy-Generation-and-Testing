#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA200 trend filter + volume spike confirmation
# - Primary signal: Donchian channel breakout (20-period high/low) on 4h timeframe
# - Trend filter: 1d EMA200 - ensures alignment with higher timeframe trend (bull/bear agnostic)
# - Volume confirmation: 4h volume > 1.5x 20-period median volume (avoid low-participation breakouts)
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian captures breakouts, EMA200 filter avoids counter-trend trades

name = "4h_1d_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Pre-compute 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume regime: volume > 1.5x 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR price crosses below 1d EMA200
            if close[i] < donchian_low[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR price crosses above 1d EMA200
            if close[i] > donchian_high[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and 1d EMA200 filter
            # Long: price breaks above Donchian high AND volume regime AND price above 1d EMA200
            if (close[i] > donchian_high[i] and 
                volume_regime[i] and 
                close[i] > ema_200_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume regime AND price below 1d EMA200
            elif (close[i] < donchian_low[i] and 
                  volume_regime[i] and 
                  close[i] < ema_200_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals