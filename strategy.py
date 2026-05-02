#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation
# Uses 4h primary timeframe for Donchian channel breakout signals
# 12h HMA(21) confirms medium-term trend direction (avoids counter-trend trades)
# Volume confirmation (2.0x 20-period average) ensures strong participation
# Discrete position sizing (0.30) balances profit potential with fee drag minimization
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian provides clear structure, HMA adds trend filter, volume confirms conviction
# Works in both bull and bear markets by only trading in direction of 12h trend

name = "4h_Donchian20_12hHMA21_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA(21)
    close_12h = pd.Series(df_12h['close'])
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    wma_half = close_12h.rolling(window=half_length, min_periods=half_length).mean()
    wma_full = close_12h.rolling(window=21, min_periods=21).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_12h = raw_hma.rolling(window=sqrt_length, min_periods=sqrt_length).mean().values
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper_channel = high_ma.shift(1).values  # Use previous bar to avoid look-ahead
    lower_channel = low_ma.shift(1).values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Donchian breakout long: price > upper channel
            # Donchian breakout short: price < lower channel
            breakout_long = close[i] > upper_channel[i]
            breakout_short = close[i] < lower_channel[i]
            
            # 12h HMA trend filter: price > HMA for longs, price < HMA for shorts
            hma_long = close[i] > hma_12h_aligned[i]
            hma_short = close[i] < hma_12h_aligned[i]
            
            if breakout_long and hma_long and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            elif breakout_short and hma_short and volume_spike[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Donchian breakdown (price < lower channel) or trend reversal
            if close[i] < lower_channel[i] or close[i] < hma_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Donchian breakout (price > upper channel) or trend reversal
            if close[i] > upper_channel[i] or close[i] > hma_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals