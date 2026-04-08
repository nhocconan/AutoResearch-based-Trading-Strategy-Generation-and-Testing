#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend and volume confirmation
# Uses Camarilla levels from 1d data for entry, 1d EMA for trend filter, and volume spike for confirmation
# Designed to work in both bull and bear markets by requiring strong trend alignment and volume confirmation
# Target: 12-37 trades/year, focused on high-probability breakouts with confirmation
name = "12h_camarilla_pivot_breakout_1d_trend_volume_v1"
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
    
    # Get 1d data for Camarilla calculation and trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA for trend filter (50-period)
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume SMA for volume context (20-period)
    vol_sma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from 1d data
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1/12
    # R2 = C + (H - L) * 1.1/6
    # R3 = C + (H - L) * 1.1/4
    # R4 = C + (H - L) * 1.1/2
    # S1 = C - (H - L) * 1.1/12
    # S2 = C - (H - L) * 1.1/6
    # S3 = C - (H - L) * 1.1/4
    # S4 = C - (H - L) * 1.1/2
    pivot_point = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    r4 = close_1d + range_hl * 1.1 / 2.0
    r3 = close_1d + range_hl * 1.1 / 4.0
    r2 = close_1d + range_hl * 1.1 / 6.0
    r1 = close_1d + range_hl * 1.1 / 12.0
    s1 = close_1d - range_hl * 1.1 / 12.0
    s2 = close_1d - range_hl * 1.1 / 6.0
    s3 = close_1d - range_hl * 1.1 / 4.0
    s4 = close_1d - range_hl * 1.1 / 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d[i]) or np.isnan(volume_1d[i]) or 
            np.isnan(vol_sma_1d[i]) or np.isnan(pivot_point[i]) or 
            np.isnan(r4[i]) or np.isnan(r3[i]) or np.isnan(r2[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(s2[i]) or 
            np.isnan(s3[i]) or np.isnan(s4[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d values for current 12h bar
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)[i]
        vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)[i]
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)[i]
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)[i]
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)[i]
        r2_aligned = align_htf_to_ltf(prices, df_1d, r2)[i]
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)[i]
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)[i]
        s2_aligned = align_htf_to_ltf(prices, df_1d, s2)[i]
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)[i]
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)[i]
        
        # Trend filter: price above/below 50 EMA on 1d
        uptrend = close[i] > ema_1d_aligned
        downtrend = close[i] < ema_1d_aligned
        
        # Volume filter: current volume above 2.5x 1d average volume (more selective)
        volume_filter = volume[i] > (vol_sma_1d_aligned * 2.5)
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 OR trend reversal
            if close[i] < s3_aligned or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR trend reversal
            if close[i] > r3_aligned or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long: price breaks above R3 + uptrend + volume filter
            if close[i] > r3_aligned and uptrend and volume_filter:
                position = 1
                signals[i] = 0.30
            # Short: price breaks below S3 + downtrend + volume filter
            elif close[i] < s3_aligned and downtrend and volume_filter:
                position = -1
                signals[i] = -0.30
    
    return signals