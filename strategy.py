#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter with volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trending vs ranging markets
# Only trade when Alligator is "awake" (JAW > TEETH > LIPS for uptrend, reverse for downtrend)
# 1d EMA50 determines primary trend - multi-timeframe alignment with daily trend
# Volume spike (1.8x 20-period average) ensures strong participation
# Discrete position sizing (0.25) minimizes fee drag
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Alligator works in both bull/bear markets by filtering for trending conditions only

name = "6h_WilliamsAlligator_1dEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(50) for trend determination
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h timeframe
    # Jaw (Blue): 13-period SMMA, smoothed 8 bars ahead
    # Teeth (Red): 8-period SMMA, smoothed 5 bars ahead  
    # Lips (Green): 5-period SMMA, smoothed 3 bars ahead
    jaw = pd.Series(high).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean().shift(8).values
    
    teeth = pd.Series(low).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean().shift(5).values
    
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean().shift(3).values
    
    # Alligator conditions: awake and trending
    alligator_long = (jaw > teeth) & (teeth > lips)  # Bullish alignment
    alligator_short = (jaw < teeth) & (teeth < lips)  # Bearish alignment
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator bullish + volume spike + close > 1d EMA50 (bullish trend)
            if alligator_long[i] and volume_spike[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + volume spike + close < 1d EMA50 (bearish trend)
            elif alligator_short[i] and volume_spike[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish or close < 1d EMA50 (trend reversal)
            if not alligator_long[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish or close > 1d EMA50 (trend reversal)
            if not alligator_short[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals