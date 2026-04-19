#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d ATR filter + volume confirmation
# Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends
# When lines are intertwined (no trend), stay out; when aligned, trade in direction
# 1d ATR filter avoids trading in low volatility conditions
# Volume confirmation ensures participation
# Designed for 50-150 total trades over 4 years (12-37/year)
name = "6h_Alligator_1dATR_Volume"
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
    
    # 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_1d_ma = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_ma)
    
    # Alligator on 6h: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Alligator alignment: jaws < teeth < lips for down, jaws > teeth > lips for up
    ma_aligned_up = (jaw > teeth) & (teeth > lips)
    ma_aligned_down = (jaw < teeth) & (teeth < lips)
    
    # Volume confirmation: volume > 1.2 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ATR indicates sufficient volatility (> 80% of MA)
        vol_filter = atr_1d[i] > (atr_1d_aligned[i] * 0.8)
        
        if position == 0:
            # Long: Alligator aligned up + volume + volatility filter
            if (ma_aligned_up[i] and volume_confirm[i] and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down + volume + volatility filter
            elif (ma_aligned_down[i] and volume_confirm[i] and vol_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator alignment breaks down
            if not ma_aligned_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator alignment breaks up
            if not ma_aligned_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals