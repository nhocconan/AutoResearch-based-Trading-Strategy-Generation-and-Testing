#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Long when Jaw < Teeth < Lips (bullish alignment) with volume > 1.8x 28-bar average and close > 1d EMA50 (uptrend)
# Short when Jaw > Teeth > Lips (bearish alignment) with volume > 1.8x 28-bar average and close < 1d EMA50 (downtrend)
# Exit when Alligator lines cross (alignment breaks) or volume drops below average
# Williams Alligator identifies trend phases and works in both bull and bear markets by capturing strong directional moves.
# Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete sizing (0.25) to minimize fee churn.

name = "12h_Williams_Alligator_1dEMA50_Volume_v1"
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator (13,8,5 smoothed with 8,5,3 periods)
    # Jaw: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods  
    # Lips: 5-period SMMA smoothed by 3 periods
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values
    
    # Volume confirmation (1.8x 28-period average)
    vol_ma = pd.Series(volume).rolling(window=28, min_periods=28).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(13, 8, 5, 28) + 8  # SMMA periods + smoothing + volume MA + shift
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Jaw < Teeth < Lips (bullish alignment) with volume spike and close > 1d EMA50 (uptrend)
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and 
                volume_spike[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Jaw > Teeth > Lips (bearish alignment) with volume spike and close < 1d EMA50 (downtrend)
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  volume_spike[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bullish alignment breaks (Jaw >= Teeth or Teeth >= Lips)
            if jaw[i] >= teeth[i] or teeth[i] >= lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bearish alignment breaks (Jaw <= Teeth or Teeth <= Lips)
            if jaw[i] <= teeth[i] or teeth[i] <= lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals