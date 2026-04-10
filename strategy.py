#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high in 12h uptrend (close > EMA50) with volume > 2.0x 20-bar avg
# - Short when price breaks below Donchian(20) low in 12h downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - 12h trend filter reduces false signals in counter-trend markets
# - Donchian breakout provides clear structure-based entries
# - Volume confirmation ensures institutional participation

name = "4h_12h_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h volume confirmation: > 2.0x 20-period average
    avg_volume_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (2.0 * avg_volume_20_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Pre-compute Donchian channels on 4h data
    highest_high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    lowest_low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_spike_12h_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Donchian breakout above in 12h uptrend with volume spike
            if (prices['close'].iloc[i] > highest_high_20[i] and 
                prices['close'].iloc[i] > ema_50_12h_aligned[i] and 
                vol_spike_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Donchian breakout below in 12h downtrend with volume spike
            elif (prices['close'].iloc[i] < lowest_low_20[i] and 
                  prices['close'].iloc[i] < ema_50_12h_aligned[i] and 
                  vol_spike_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit long when price returns to midpoint of Donchian channel
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2.0
            if position == 1 and prices['close'].iloc[i] < midpoint:
                position = 0
                signals[i] = 0.0
            # Exit short when price returns to midpoint
            elif position == -1 and prices['close'].iloc[i] > midpoint:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals