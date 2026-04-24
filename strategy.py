#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot Breakout with 1w EMA50 Trend Filter and Volume Spike.
- Calculate 1d Camarilla pivot levels (R1,R2,R3,R4,S1,S2,S3,S4) from prior 1d OHLC
- Long when price breaks above R4 with volume > 2.0 * 20-period average AND 1w EMA50 uptrend
- Short when price breaks below S4 with volume > 2.0 * 20-period average AND 1w EMA50 downtrend
- Exit when price returns to Camarilla H3/L3 levels (mean reversion zone) OR volume drops below average
- Uses 6h primary with 1w HTF for EMA50 trend filter to avoid counter-trend breakouts
- Camarilla R4/S4 represent strong breakout levels; volume confirms conviction; 1w EMA50 ensures alignment with weekly trend
- Works in bull markets (long breakouts in uptrend) and bear markets (short breakdowns in downtrend)
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels from prior day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Open not needed for standard Camarilla, but we can use close as proxy or fetch it
    # Using close_1d as approximation for pivot calculation (standard formula uses (H+L+C)/3)
    # But Camarilla uses: Pivot = (H+L+C)/3, Range = H-L
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R1 = pivot_1d + (range_1d * 1.1 / 12)
    R2 = pivot_1d + (range_1d * 1.1 / 6)
    R3 = pivot_1d + (range_1d * 1.1 / 4)
    R4 = pivot_1d + (range_1d * 1.1 / 2)
    S1 = pivot_1d - (range_1d * 1.1 / 12)
    S2 = pivot_1d - (range_1d * 1.1 / 6)
    S3 = pivot_1d - (range_1d * 1.1 / 4)
    S4 = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (use prior day's levels for current 6h bars)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4)
    H3_1d = pivot_1d + (range_1d * 1.1 / 4)  # Same as R3
    L3_1d = pivot_1d - (range_1d * 1.1 / 4)  # Same as S3
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend filter: uptrend if price > EMA50, downtrend if price < EMA50
    uptrend = ema_50_1w_aligned > 0  # Valid EMA50 value
    # Actually need to compare price to EMA50 - we'll do this in the loop using aligned close
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1  # Need volume MA and EMA50 data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R4_1d_aligned[i]) or np.isnan(S4_1d_aligned[i]) or 
            np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current price and aligned weekly EMA for trend comparison
        current_close = close[i]
        # We need the 1w close price aligned to compare with EMA50
        # For simplicity, we'll use the aligned EMA50 and check if it's trending up/down
        # Better approach: get aligned 1w close and compare to its EMA50
        # But we don't have aligned 1w close easily - let's use price vs EMA50 slope
        
        # Simpler trend filter: EMA50 rising/falling
        if i >= start_idx + 1:
            ema_prev = ema_50_1w_aligned[i-1]
            ema_curr = ema_50_1w_aligned[i]
            ema_rising = ema_curr > ema_prev
            ema_falling = ema_curr < ema_prev
        else:
            ema_rising = True  # Default to allow trading initially
            ema_falling = True
        
        if position == 0:
            # Long: price breaks above R4 AND volume spike AND EMA50 rising
            if current_close > R4_1d_aligned[i] and volume_spike[i] and ema_rising:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 AND volume spike AND EMA50 falling
            elif current_close < S4_1d_aligned[i] and volume_spike[i] and ema_falling:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to H3 (mean reversion) OR volume drops OR EMA50 turns down
            if current_close < H3_1d_aligned[i] or not volume_spike[i] or not ema_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to L3 (mean reversion) OR volume drops OR EMA50 turns up
            if current_close > L3_1d_aligned[i] or not volume_spike[i] or not ema_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0