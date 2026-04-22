#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w EMA100 trend filter and volume spike
    # Donchian channels provide clear breakout levels based on 20-day highs/lows
    # 1w EMA100 filters for long-term trend direction to avoid counter-trend trades
    # Volume spike (2x 20-day average) confirms institutional participation
    # Works in bull/bear: breaks through key levels with trend and volume confirmation
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian(20) calculation
    df_1d = get_htf_data(prices, '1d')
    # Calculate 20-period high and low for Donchian channels
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (using previous day's levels)
    # Donchian breakout uses the channel from 20 periods ago, so we shift by 1
    donchian_high = np.roll(high_20, 1)
    donchian_low = np.roll(low_20, 1)
    donchian_high[0] = high_20[0]  # first value
    donchian_low[0] = low_20[0]
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Load 1w data for EMA100 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema100_1w = pd.Series(df_1w['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema100_1w)
    
    # Volume spike filter (20-period on 1d)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema100_1w_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume spike and price above 1w EMA100 (uptrend)
            if close[i] > donchian_high_aligned[i] and vol_spike[i] and close[i] > ema100_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume spike and price below 1w EMA100 (downtrend)
            elif close[i] < donchian_low_aligned[i] and vol_spike[i] and close[i] < ema100_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian level (low for longs, high for shorts)
            if position == 1:
                if close[i] < donchian_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_1wEMA100_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0