#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 12h EMA20 trend + volume spike
    # Donchian breakouts capture breakouts from key support/resistance levels
    # 12h EMA20 provides medium-term trend filter to avoid counter-trend trades
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Works in bull/bear: breaks through key levels with trend and volume confirmation
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA20 trend
    df_12h = get_htf_data(prices, '12h')
    ema20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema20_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian with volume spike and price above 12h EMA20 (uptrend)
            if close[i] > highest_high[i] and vol_spike[i] and close[i] > ema20_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian with volume spike and price below 12h EMA20 (downtrend)
            elif close[i] < lowest_low[i] and vol_spike[i] and close[i] < ema20_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian level (lower for longs, upper for shorts)
            if position == 1:
                if close[i] < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_Breakout_12hEMA20_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0