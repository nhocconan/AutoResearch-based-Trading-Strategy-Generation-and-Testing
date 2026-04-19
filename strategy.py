#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w volume confirmation and 1d trend filter
# - Camarilla levels (R1/S1, R2/S2, R3/S3, R4/S4) calculated from previous 1d candle
# - Long breakout: price closes above R3 with volume > 1.5x 1w average volume
# - Short breakdown: price closes below S3 with volume > 1.5x 1w average volume
# - Trend filter: only take longs when price > 1d EMA50, shorts when price < 1d EMA50
# - Exit when price returns to the 1d close (pivot) or opposite Camarilla level is touched
# - Designed to capture institutional breakouts in both bull and bear markets
# - Target: 15-30 trades/year to minimize fee drag while capturing strong moves

name = "6h_Camarilla_R3S3_Breakout_1dTrend_1wVolume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d candle
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # where C = (H+L+C)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    high_low = df_1d['high'] - df_1d['low']
    
    r1 = typical_price + high_low * 1.1 / 12
    r2 = typical_price + high_low * 1.1 / 6
    r3 = typical_price + high_low * 1.1 / 4
    r4 = typical_price + high_low * 1.1 / 2
    s1 = typical_price - high_low * 1.1 / 12
    s2 = typical_price - high_low * 1.1 / 6
    s3 = typical_price - high_low * 1.1 / 4
    s4 = typical_price - high_low * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, typical_price.values)  # Pivot point
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    
    # 1w volume average (20-period)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x 1w average volume (scaled)
        # Scale 1w average to 6h: 1w has 28x 6h bars (7 days * 4 per day), so divide by 28
        volume_filter = vol_ma_1w_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1w_aligned[i] / 28.0)
        
        if position == 0:
            # Look for long breakout: price closes above R3 + volume + uptrend
            if close[i] > r3_aligned[i] and volume_filter and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Look for short breakdown: price closes below S3 + volume + downtrend
            elif close[i] < s3_aligned[i] and volume_filter and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to pivot or breaks S1 (mean reversion)
            if close[i] <= pivot_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to pivot or breaks R1 (mean reversion)
            if close[i] >= pivot_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals