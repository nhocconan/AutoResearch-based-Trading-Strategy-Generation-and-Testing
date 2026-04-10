#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) crosses above -80 (oversold) AND 1d close > 1d EMA50 AND volume > 1.5x average
# - Short when Williams %R(14) crosses below -20 (overbought) AND 1d close < 1d EMA50 AND volume > 1.5x average
# - Exit when Williams %R returns to -50 (mean reversion) OR volume drops below 0.7x average
# - Uses 1d trend filter to avoid counter-trend trades and volume spike to confirm momentum
# - Williams %R is effective in ranging markets (2025+) and catches reversals in trends
# - Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag

name = "6h_1d_williamsr_meanreversion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute Williams %R(14) on 6h timeframe
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * (highest_high - close_6h) / (highest_high - lowest_low), 
                          -50)  # Default to neutral when range is zero
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(high_6h[i]) or 
            np.isnan(low_6h[i]) or np.isnan(close_6h[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long entry: Williams %R crosses above -80 from below (oversold bounce)
            # WITH 1d uptrend filter AND volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and  # Cross above -80
                prices['close'].iloc[i] > ema50_1d_aligned[i] and   # 1d uptrend
                vol_spike.iloc[i]):                               # Volume confirmation
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R crosses below -20 from above (overbought rejection)
            # WITH 1d downtrend filter AND volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and  # Cross below -20
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and   # 1d downtrend
                  vol_spike.iloc[i]):                               # Volume confirmation
                position = -1
                signals[i] = -0.25
        
        elif position == 1:  # Long position - look for exit
            # Exit conditions:
            # 1. Williams %R returns to -50 (mean reversion achieved)
            # 2. Volume drops below 0.7x average (loss of momentum)
            if (williams_r[i] >= -50 or vol_weak.iloc[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Hold long
        
        elif position == -1:  # Short position - look for exit
            # Exit conditions:
            # 1. Williams %R returns to -50 (mean reversion achieved)
            # 2. Volume drops below 0.7x average (loss of momentum)
            if (williams_r[i] <= -50 or vol_weak.iloc[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Hold short
    
    return signals