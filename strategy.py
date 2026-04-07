#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Williams Alligator + Daily Trend + Volume Spike
# Hypothesis: Williams Alligator (three SMAs) identifies trend direction and momentum.
# Combined with daily trend filter and volume confirmation, it captures strong trends
# while avoiding whipsaws. Works in both bull/bear by using Alligator's alignment.
# Target: 15-25 trades/year (60-100 total).

name = "12h_williams_alligator_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: three SMAs (Jaw, Teeth, Lips)
    # Jaw: 13-period SMA, shifted 8 bars forward
    # Teeth: 8-period SMA, shifted 5 bars forward  
    # Lips: 5-period SMA, shifted 3 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: Alligator lines converge (lips < teeth) or trend turns bearish
            if lips[i] < teeth[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Alligator lines converge (lips > teeth) or trend turns bullish
            if lips[i] > teeth[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Alligator aligned up: lips > teeth > jaw (bullish)
                if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Alligator aligned down: lips < teeth < jaw (bearish)
                elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals