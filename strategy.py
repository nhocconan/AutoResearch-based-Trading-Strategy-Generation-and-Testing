#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion + 1d HMA(34) trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# 1d HMA(34) ensures trades align with medium-term trend to avoid counter-trend losses
# Volume confirmation (1.8x 20-period average) filters low-conviction breakouts
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in bull markets by buying dips in uptrends, in bear markets by selling rallies in downtrends

name = "12h_WilliamsR_1dHMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d HMA(34)
    close_1d = pd.Series(df_1d['close'])
    half_length = 34 // 2
    sqrt_length = int(np.sqrt(34))
    
    wma_half = close_1d.rolling(window=half_length, min_periods=half_length).mean()
    wma_full = close_1d.rolling(window=34, min_periods=34).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_1d = raw_hma.rolling(window=sqrt_length, min_periods=sqrt_length).mean().values
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    williams_r = williams_r.values
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(hma_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Williams %R long: oversold (< -80) in uptrend (price > HMA)
            # Williams %R short: overbought (> -20) in downtrend (price < HMA)
            long_condition = williams_r[i] < -80 and close[i] > hma_1d_aligned[i]
            short_condition = williams_r[i] > -20 and close[i] < hma_1d_aligned[i]
            
            if long_condition and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_condition and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R overbought (> -20) or trend reversal
            if williams_r[i] > -20 or close[i] < hma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R oversold (< -80) or trend reversal
            if williams_r[i] < -80 or close[i] > hma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals