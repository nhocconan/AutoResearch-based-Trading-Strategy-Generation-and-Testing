#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R with EMA trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions that work in both bull and bear markets
# EMA(50) on 1d timeframe provides trend filter: only take long signals in uptrend, short in downtrend
# Volume confirmation (current 6h volume > 1.5x 20-period average) filters low-quality signals
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_williamsr_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Williams %R and EMA to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        # Trend filter: determine market direction from 1d EMA
        # Uptrend: price above EMA50, Downtrend: price below EMA50
        is_uptrend = close_1d[-1] > ema_50[-1] if len(close_1d) > 0 else True  # Simplified: use current trend
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when Williams %R rises above -20 (overbought) or reverse signal
            if williams_r_aligned[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -80 (oversold) or reverse signal
            if williams_r_aligned[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Williams %R signals with volume confirmation and trend filter
            # Long: Williams %R crosses below -80 (oversold) in uptrend
            # Short: Williams %R crosses above -20 (overbought) in downtrend
            if i > 0:
                prev_williams = williams_r_aligned[i-1]
                curr_williams = williams_r_aligned[i]
                
                # Long signal: Williams %R crosses above -80 from below (exit oversold)
                if prev_williams <= -80 and curr_williams > -80:
                    # Only take long in uptrend
                    if close[i] > ema_50_aligned[i]:
                        position = 1
                        signals[i] = position_size
                # Short signal: Williams %R crosses below -20 from above (exit overbought)
                elif prev_williams >= -20 and curr_williams < -20:
                    # Only take short in downtrend
                    if close[i] < ema_50_aligned[i]:
                        position = -1
                        signals[i] = -position_size
    
    return signals