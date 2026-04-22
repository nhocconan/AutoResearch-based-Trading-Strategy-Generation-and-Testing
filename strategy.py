#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + volume spike + EMA50 trend filter
    # Donchian channels capture volatility-based support/resistance
    # Volume spike confirms institutional participation
    # EMA50 filters for medium-term trend to avoid counter-trend trades
    # Works in bull/bear: breaks through volatility bands with trend and volume
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for EMA50 trend (trend filter on same timeframe)
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Donchian channels (20-period) on 4h timeframe
    # Use rolling window on high/low with proper alignment
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume spike and price above EMA50 (uptrend)
            if close[i] > donchian_high[i] and vol_spike[i] and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume spike and price below EMA50 (downtrend)
            elif close[i] < donchian_low[i] and vol_spike[i] and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian level (lower band for longs, upper band for shorts)
            if position == 1:
                if close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_EMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0