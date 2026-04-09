#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + volume confirmation + weekly trend filter (EMA34)
# - Primary signal: 1d price breaks above/below 20-period Donchian channel
# - Volume confirmation: 1d volume > 20-period median volume (avoid low-participation breakouts)
# - Weekly trend filter: Only trade in direction of weekly EMA34 (avoid counter-trend in strong weekly trends)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 30-100 total trades over 4 years (7-25/year) per 1d strategy guidelines
# - Works in bull/bear: Donchian captures structure, weekly EMA filter adapts to higher timeframe trend

name = "1d_1w_donchian_vol_trend_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute weekly indicators
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to daily timeframe (completed weekly bar only)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channel (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d volume regime: volume > 20-period median volume (avoid low participation)
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if close[i] <= lowest_low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if close[i] >= highest_high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and weekly trend filter
            # Long: price breaks above Donchian upper band AND volume regime AND price above weekly EMA34
            if high[i] >= highest_high_20[i] and volume_regime[i] and close[i] > ema_34_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower band AND volume regime AND price below weekly EMA34
            elif low[i] <= lowest_low_20[i] and volume_regime[i] and close[i] < ema_34_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals