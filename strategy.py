#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends
# In bull markets: buy when Lips > Teeth > Jaw (aligned up) + price above 1d EMA50 + volume spike
# In bear markets: sell when Lips < Teeth < Jaw (aligned down) + price below 1d EMA50 + volume spike
# Volume spike (>2.0x 50-period EMA) confirms institutional participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Jaw(13), Teeth(8), Lips(5) - all SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # Red line
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # Green line
    
    # Volume confirmation: 50-period EMA on volume
    vol_series = pd.Series(volume)
    vol_ema_50 = vol_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start from 60 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 50-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_50[i])
        
        # Williams Alligator signals with 1d trend filter
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_aligned = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_aligned = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            if bullish_aligned and close[i] > ema_50_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif bearish_aligned and close[i] < ema_50_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment OR price below 1d EMA50
            if bearish_aligned or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment OR price above 1d EMA50
            if bullish_aligned or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals