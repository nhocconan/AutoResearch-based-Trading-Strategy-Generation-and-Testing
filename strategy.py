#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation
# Uses 1d primary timeframe for Donchian breakout signals
# 1w HMA(21) confirms longer-term trend direction (avoids counter-trend trades)
# Volume confirmation (1.8x 20-period average) ensures strong participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Donchian provides clear structure, 1w HMA adds robust trend filter, volume confirms conviction
# Works in both bull and bear markets by only trading in direction of 1w trend

name = "1d_Donchian20_1wHMA21_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1w HMA(21)
    close_1w = pd.Series(df_1w['close'])
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    wma_half = close_1w.rolling(window=half_length, min_periods=half_length).mean()
    wma_full = close_1w.rolling(window=21, min_periods=21).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_1w = raw_hma.rolling(window=sqrt_length, min_periods=sqrt_length).mean().values
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(hma_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Donchian breakout long: price > upper channel
            # Donchian breakout short: price < lower channel
            breakout_long = close[i] > high_ma[i]
            breakout_short = close[i] < low_ma[i]
            
            # 1w HMA trend filter: price > HMA for longs, price < HMA for shorts
            hma_long = close[i] > hma_1w_aligned[i]
            hma_short = close[i] < hma_1w_aligned[i]
            
            if breakout_long and hma_long and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif breakout_short and hma_short and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Donchian breakdown (price < lower channel) or trend reversal
            if close[i] < low_ma[i] or close[i] < hma_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Donchian breakout (price > upper channel) or trend reversal
            if close[i] > high_ma[i] or close[i] > hma_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals