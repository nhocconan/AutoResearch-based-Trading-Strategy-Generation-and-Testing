#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA trend filter + volume confirmation
# - Williams %R(14) on 6h for overbought/oversold signals (< -80 long, > -20 short)
# - 1d EMA(50) as trend filter: only long when price > EMA50, short when price < EMA50
# - 6h volume > 1.5x 20-period average for confirmation
# - Position size: 0.25 (25% of capital)
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Williams %R is effective in ranging markets; EMA filter avoids counter-trend trades
# - Volume confirmation reduces false signals
# - Works in both bull and bear markets via trend filter

name = "6h_1d_williamsr_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # 6h volume > 1.5x 20-period average (volume confirmation)
    volume = prices['volume'].values
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or
            ema_50_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Williams %R rises above -50 (exiting oversold) or volume drops
            if williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -50 (exiting overbought) or volume drops
            if williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with trend and volume confirmation
            if (williams_r[i] <= -80 and    # Oversold
                close[i] > ema_50_1d_aligned[i] and  # Uptrend filter (price above 1d EMA50)
                volume_spike[i]):           # Volume confirmation
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] >= -20 and   # Overbought
                  close[i] < ema_50_1d_aligned[i] and  # Downtrend filter (price below 1d EMA50)
                  volume_spike[i]):          # Volume confirmation
                position = -1
                signals[i] = -0.25
    
    return signals