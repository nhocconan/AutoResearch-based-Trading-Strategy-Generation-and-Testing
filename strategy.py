#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R with 1w EMA200 Trend Filter and Volume Spike.
Long when Williams %R < -80 (oversold) AND price > 1w EMA200 (long-term uptrend) AND volume > 1.5 * 20-period average volume.
Short when Williams %R > -20 (overbought) AND price < 1w EMA200 (long-term downtrend) AND volume > 1.5 * 20-period average volume.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) or weekly trend reverses.
Uses 1d for Williams %R calculation, 1w for EMA200 trend filter.
Target: 30-100 total trades over 4 years (7-25/year). Williams %R captures mean reversion in extremes, 
weekly EMA200 filters for higher-timeframe trend alignment to reduce false signals in chop, volume spike confirms conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 1d timeframe (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 20-period average volume for volume spike confirmation
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Williams %R to 1d timeframe (no alignment needed as we're already on 1d)
    williams_r_aligned = williams_r  # Already on 1d timeframe
    
    # Align 1w EMA200 to 1d timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Align 20-period average volume to 1d timeframe (no alignment needed)
    avg_volume_20_aligned = avg_volume_20  # Already on 1d timeframe
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(williams_r_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or np.isnan(avg_volume_20_aligned[i]):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        price = close[i]
        ema200 = ema200_1w_aligned[i]
        vol = volume[i]
        avg_vol = avg_volume_20_aligned[i]
        
        # Volume spike condition: current volume > 1.5 * 20-period average volume
        volume_spike = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1w EMA200 (long-term uptrend) AND volume spike
            if wr < -80 and price > ema200 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1w EMA200 (long-term downtrend) AND volume spike
            elif wr > -20 and price < ema200 and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR price < 1w EMA200 (trend reversal)
            if wr > -50 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR price > 1w EMA200 (trend reversal)
            if wr < -50 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_WeeklyEMA200_VolumeSpike"
timeframe = "1d"
leverage = 1.0