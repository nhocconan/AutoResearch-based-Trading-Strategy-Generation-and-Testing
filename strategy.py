#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1d EMA100 trend filter and volume spike
    # Donchian channels provide clear trend-following breakout signals
    # 1d EMA100 filters for long-term trend direction to avoid counter-trend trades
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Works in bull/bear: breaks through key levels with trend and volume confirmation
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA100 trend
    df_1d = get_htf_data(prices, '1d')
    ema100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Donchian channel parameters (20-period on 12h timeframe)
    donchian_period = 20
    
    # Calculate Donchian channels
    high_roll = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    low_roll = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema100_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian with volume spike and price above 1d EMA100 (uptrend)
            if close[i] > donchian_upper[i] and vol_spike[i] and close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian with volume spike and price below 1d EMA100 (downtrend)
            elif close[i] < donchian_lower[i] and vol_spike[i] and close[i] < ema100_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian level (lower for longs, upper for shorts)
            if position == 1:
                if close[i] < donchian_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dEMA100_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0