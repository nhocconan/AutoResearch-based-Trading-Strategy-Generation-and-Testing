#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d/1w trend filter and volume confirmation
# Uses Williams Alligator (jaw/teeth/lips) to identify trendless markets and avoid whipsaws
# Requires price outside Alligator mouth + 1d EMA50 trend alignment + volume spike filter
# Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe
# Proven pattern: Alligator + volume + trend filter works across BTC/ETH in all regimes

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Alligator on 12h timeframe (using median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume spike filter (volume > 1.5x 20-period average)
    volume_1d_series = pd.Series(volume_1d)
    avg_volume_20 = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * avg_volume_20)
    
    # Calculate 1w EMA50 for stronger trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator mouth condition: lips outside jaw/teeth (trending market)
        lips_above_jaw = lips_aligned[i] > jaw_aligned[i]
        lips_below_jaw = lips_aligned[i] < jaw_aligned[i]
        teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
        teeth_below_jaw = teeth_aligned[i] < jaw_aligned[i]
        
        # Strong uptrend: lips > teeth > jaw (Alligator opening up)
        strong_uptrend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Strong downtrend: lips < teeth < jaw (Alligator opening down)
        strong_downtrend = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long entry: strong uptrend + price above Alligator + 1d EMA50 uptrend + volume spike
            if (strong_uptrend and 
                close[i] > lips_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: strong downtrend + price below Alligator + 1d EMA50 downtrend + volume spike
            elif (strong_downtrend and 
                  close[i] < lips_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator starts closing (lips < teeth) or price breaks below teeth
            if lips_aligned[i] < teeth_aligned[i] or close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator starts closing (lips > teeth) or price breaks above teeth
            if lips_aligned[i] > teeth_aligned[i] or close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals