#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-week pivot breakout with volume and volatility filter
# - Go long when price breaks above weekly pivot resistance (R1) on 12h + volume > 1.5x 20-period average + ATR(12h) < 1.5x ATR(360h) (low vol regime)
# - Go short when price breaks below weekly pivot support (S1) on 12h + volume > 1.5x 20-period average + ATR(12h) < 1.5x ATR(360h)
# - Exit when price crosses back through weekly pivot point (PP) or volatility increases
# - Uses 1w for pivot calculation (more stable in ranging/trending markets) and 12h for entry/exit timing
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    
    # Align weekly pivots to 12h timeframe
    pp_12h = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_12h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_12h = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate ATR on 12h timeframe for volatility filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr12 = pd.Series(tr).rolling(window=12, min_periods=12).mean().values
    atr360 = pd.Series(tr).rolling(window=360, min_periods=360).mean().values  # 30 weeks of 12h bars
    
    # Volatility filter: low volatility regime (ATR12 < 1.5 * ATR360)
    vol_filter = atr12 < 1.5 * atr360
    
    # Volume confirmation
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(360, n):  # Start after ATR360 warmup
        # Skip if NaN in indicators
        if np.isnan(pp_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or \
           np.isnan(atr12[i]) or np.isnan(atr360[i]) or np.isnan(vol_ma[i]) or np.isnan(vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above weekly R1 + low vol + volume surge
            if price > r1_12h[i] and vol_filter[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S1 + low vol + volume surge
            elif price < s1_12h[i] and vol_filter[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below weekly PP OR volatility increases
            if price < pp_12h[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above weekly PP OR volatility increases
            if price > pp_12h[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_R1S1_Breakout_VolFilter_VolumeSurge"
timeframe = "12h"
leverage = 1.0