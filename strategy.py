#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trend and convergence/divergence
# In trending markets: Jaw > Teeth > Lips (bullish) or Jaw < Teeth < Lips (bearish)
# In ranging markets: lines intertwine (Alligator sleeping)
# 1d EMA50 ensures alignment with longer-term trend to avoid counter-trend trades
# Volume spike (2.0x 20-bar MA) confirms institutional participation
# Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# Works in bull markets (trend following with Alligator alignment) and bear markets (mean reversion when Alligator sleeps)

name = "6h_Williams_Alligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h timeframe (Jaw=13, Teeth=8, Lips=5)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator: Jaw > Teeth > Lips (trending up)
            bullish_alligator = jaw[i] > teeth[i] and teeth[i] > lips[i]
            # Bearish Alligator: Jaw < Teeth < Lips (trending down)
            bearish_alligator = jaw[i] < teeth[i] and teeth[i] < lips[i]
            
            # Long entry: Bullish Alligator AND price > 1d EMA50 AND volume spike
            if (bullish_alligator and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish Alligator AND price < 1d EMA50 AND volume spike
            elif (bearish_alligator and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator sleeps (lines intertwine) OR price below 1d EMA50 (trend change)
            # Alligator sleeping: not (Jaw > Teeth > Lips) and not (Jaw < Teeth < Lips)
            alligator_sleeping = not (jaw[i] > teeth[i] and teeth[i] > lips[i]) and not (jaw[i] < teeth[i] and teeth[i] < lips[i])
            if alligator_sleeping or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator sleeps (lines intertwine) OR price above 1d EMA50 (trend change)
            alligator_sleeping = not (jaw[i] > teeth[i] and teeth[i] > lips[i]) and not (jaw[i] < teeth[i] and teeth[i] < lips[i])
            if alligator_sleeping or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals