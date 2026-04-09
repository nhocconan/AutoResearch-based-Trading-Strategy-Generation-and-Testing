#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation
# - Primary signal: Donchian channel breakout on 1d timeframe - long when price > upper band, short when price < lower band
# - Trend filter: 1w EMA50 - ensures alignment with weekly trend (avoid counter-trend trades)
# - Volume confirmation: 1d volume > 20-period median volume (avoid low-participation breakouts)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Donchian breakouts capture strong moves, EMA50 filter avoids whipsaws in ranging markets

name = "1d_1w_donchian_ema_volume_v1"
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
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d volume regime: volume > 20-period median volume
    volume = prices['volume'].values
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
            # Exit: price crosses below Donchian middle OR weekly trend turns bearish
            middle = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close_1d[i] < middle or close_1d[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian middle OR weekly trend turns bullish
            middle = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close_1d[i] > middle or close_1d[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and 1w EMA50 filter
            # Long: price > upper Donchian band AND volume regime AND price above 1w EMA50
            if (close_1d[i] > highest_high_20[i] and 
                volume_regime[i] and 
                close_1d[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price < lower Donchian band AND volume regime AND price below 1w EMA50
            elif (close_1d[i] < lowest_low_20[i] and 
                  volume_regime[i] and 
                  close_1d[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals