#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + volume confirmation + weekly trend filter (1w EMA34)
# - Primary signal: 1d price breaks above/below 20-period Donchian channel
# - Volume confirmation: 1d volume > 20-period median volume (avoid low-participation breakouts)
# - Weekly trend filter: Only trade breakouts in direction of 1w EMA34 trend
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Donchian captures structure, weekly EMA filter adapts to higher timeframe trend

name = "1d_1w_donchian_vol_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Pre-compute weekly indicators
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to daily timeframe (completed weekly bar only)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period median volume for confirmation
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(median_volume_20[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if close[i] < lowest_low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if close[i] > highest_high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and weekly trend filter
            # Long: price breaks above upper band AND volume confirmation AND weekly uptrend
            if (high[i] > highest_high_20[i] and 
                volume[i] > median_volume_20[i] and 
                close[i] > ema_34_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower band AND volume confirmation AND weekly downtrend
            elif (low[i] < lowest_low_20[i] and 
                  volume[i] > median_volume_20[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals