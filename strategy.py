#!/usr/bin/env python3
# 4h_donchian_1d_trend_volume_v2
# Hypothesis: 4h strategy using 1d Donchian breakout for trend filter, 4h Donchian breakout for entry timing, and volume confirmation.
# In bull markets: price > 1d Donchian upper + 4h Donchian breakout + volume spike → long
# In bear markets: price < 1d Donchian lower + 4h Donchian breakdown + volume spike → short
# The 1d Donchian channel acts as a strong trend filter to avoid counter-trend trades. Volume > 1.5x 20-period average filters weak moves.
# Discrete sizing (±0.25) minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for trend filter (Donchian)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Donchian parameters
    donchian_period = 20
    
    # 1d Donchian upper and lower
    donchian_upper_1d = pd.Series(high_1d).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_lower_1d = pd.Series(low_1d).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align 1d Donchian to 4h timeframe
    donchian_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    
    # 4h Donchian for entry timing
    donchian_period_4h = 20
    donchian_upper_4h = pd.Series(high).rolling(window=donchian_period_4h, min_periods=donchian_period_4h).max().values
    donchian_lower_4h = pd.Series(low).rolling(window=donchian_period_4h, min_periods=donchian_period_4h).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_1d_aligned[i]) or np.isnan(donchian_lower_1d_aligned[i]) or
            np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian lower (trend reversal)
            if close[i] < donchian_lower_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian upper (trend reversal)
            if close[i] > donchian_upper_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price > 1d Donchian upper + 4h Donchian breakout
                if close[i] > donchian_upper_1d_aligned[i] and close[i] > donchian_upper_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price < 1d Donchian lower + 4h Donchian breakdown
                elif close[i] < donchian_lower_1d_aligned[i] and close[i] < donchian_lower_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals