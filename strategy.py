#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and 1d EMA trend filter.
# Camarilla levels provide clear support/resistance, volume confirms breakout validity,
# and 1d EMA ensures we trade with the higher timeframe trend. Designed for 20-50 trades/year
# with discrete sizing to minimize fee drag. Works in bull/bear via trend filter.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # H, L, C from daily timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ph = df_1d['high'].values  # previous day high
    pl = df_1d['low'].values   # previous day low
    pc = df_1d['close'].values # previous day close
    
    # Camarilla formula: 
    # Resistance: C + (H-L) * 1.1/12, C + (H-L) * 1.1/6, C + (H-L) * 1.1/4, C + (H-L) * 1.1/2
    # Support:    C - (H-L) * 1.1/12, C - (H-L) * 1.1/6, C - (H-L) * 1.1/4, C - (H-L) * 1.1/2
    # We use R3/S3 and R4/S4 for breakouts
    rng = ph - pl
    r3 = pc + (rng * 1.1 / 4)
    s3 = pc - (rng * 1.1 / 4)
    r4 = pc + (rng * 1.1 / 2)
    s4 = pc - (rng * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (they change only when new 1d bar forms)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: volume > 1.5x 20-period SMA
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long breakout: price closes above R3 with volume, in uptrend (price > EMA34)
            if (close[i] > r3_aligned[i] and 
                vol_filter[i] and 
                price > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S3 with volume, in downtrend (price < EMA34)
            elif (close[i] < s3_aligned[i] and 
                  vol_filter[i] and 
                  price < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below S3 (re-test of support) or against trend
            if (close[i] < s3_aligned[i] or 
                price < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above R3 (re-test of resistance) or against trend
            if (close[i] > r3_aligned[i] or 
                price > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals