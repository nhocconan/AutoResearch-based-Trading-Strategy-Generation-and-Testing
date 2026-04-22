#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian breakout with 12h EMA50 trend filter and volume confirmation
    # Donchian channels identify breakouts from price ranges
    # EMA50 on 12h filters for trend direction (works in bull/bear markets)
    # Volume spike confirms institutional participation
    # Works in both regimes: breakouts in trends, mean reversion in ranges
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian Channel (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper band + volume spike + price above EMA50 (uptrend)
            if close[i] > donchian_upper[i] and vol_spike[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band + volume spike + price below EMA50 (downtrend)
            elif close[i] < donchian_lower[i] and vol_spike[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to middle of Donchian channel
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if position == 1:
                if close[i] < donchian_middle:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_middle:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0