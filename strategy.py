#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R with 1d EMA50 Trend Filter and Volume Spike.
Long when Williams %R < -80 (oversold) AND price > 1d EMA50 (medium-term uptrend) AND volume > 1.3 * 20-period average volume.
Short when Williams %R > -20 (overbought) AND price < 1d EMA50 (medium-term downtrend) AND volume > 1.3 * 20-period average volume.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) or daily trend reverses.
Uses 12h for Williams %R calculation, 1d for EMA50 trend filter.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R captures mean reversion in extremes, 
daily EMA50 filters for medium-term trend alignment to reduce false signals in chop, volume spike confirms conviction.
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
    
    # Get 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R on 12h timeframe (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period average volume for volume spike confirmation
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h Williams %R to 12h timeframe (no alignment needed as we're already on 12h)
    williams_r_aligned = williams_r  # Already on 12h timeframe
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Align 20-period average volume to 12h timeframe (no alignment needed)
    avg_volume_20_aligned = avg_volume_20  # Already on 12h timeframe
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume_20_aligned[i]):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        price = close[i]
        ema50 = ema50_1d_aligned[i]
        vol = volume[i]
        avg_vol = avg_volume_20_aligned[i]
        
        # Volume spike condition: current volume > 1.3 * 20-period average volume
        volume_spike = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA50 (medium-term uptrend) AND volume spike
            if wr < -80 and price > ema50 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA50 (medium-term downtrend) AND volume spike
            elif wr > -20 and price < ema50 and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR price < 1d EMA50 (trend reversal)
            if wr > -50 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR price > 1d EMA50 (trend reversal)
            if wr < -50 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_DailyEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0