#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot levels from prior week as support/resistance.
# Long when price breaks above weekly R3 with volume confirmation and price above 6h EMA20.
# Short when price breaks below weekly S3 with volume confirmation and price below 6h EMA20.
# Exit when price returns to weekly pivot point or crosses EMA20 in opposite direction.
# Weekly pivots provide structure that works in both trending and ranging markets.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Resistance levels: R1 = 2*P - L, R2 = P + (H-L), R3 = H + 2*(P-L)
    r1_1w = 2 * pp_1w - low_1w
    r2_1w = pp_1w + (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pp_1w - low_1w)
    # Support levels: S1 = 2*P - H, S2 = P - (H-L), S3 = L - 2*(H-P)
    s1_1w = 2 * pp_1w - high_1w
    s2_1w = pp_1w - (high_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pp_1w)
    
    # Load daily data for EMA20 filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation: 1.5x average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 2)  # Need EMA20 and weekly data
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pp_1w_aligned[i]) or 
            np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts from weekly S3/R3
            # Long: price breaks above R3 AND above EMA20
            if (close[i] > r3_1w_aligned[i] and 
                close[i] > ema_20_1d_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below S3 AND below EMA20
            elif (close[i] < s3_1w_aligned[i] and 
                  close[i] < ema_20_1d_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly pivot or crosses below EMA20
            if (close[i] <= pp_1w_aligned[i] or 
                close[i] < ema_20_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly pivot or crosses above EMA20
            if (close[i] >= pp_1w_aligned[i] or 
                close[i] > ema_20_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WeeklyPivot_S3R3_Breakout_EMA20_v1"
timeframe = "6h"
leverage = 1.0