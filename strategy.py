#!/usr/bin/env python3
"""
12h Weekly Pivot S3/S4 Breakout with Volume Spike
Hypothesis: Weekly pivot levels act as strong support/resistance. Breaking S3 (strong support) or R3 (strong resistance) 
with volume confirmation captures significant momentum moves. Works in bull/bear markets by requiring volume confirmation 
to avoid false breakouts. Uses 12h timeframe for lower frequency and higher win rate.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: (H+L+C)/3
    # Then S3 = H - 2*(H-P), R3 = H + 2*(P-L) where P = pivot
    # Actually: S3 = Low - 2*(High - Pivot), R3 = High + 2*(Pivot - Low)
    typical_price = (df_w['high'] + df_w['low'] + df_w['close']) / 3
    pivot = typical_price.values
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    
    # Calculate S3 and R3
    s3 = weekly_low - 2 * (weekly_high - pivot)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    
    # Align to 12h timeframe (wait for weekly bar to close)
    s3_aligned = align_htf_to_ltf(prices, df_w, s3)
    r3_aligned = align_htf_to_ltf(prices, df_w, r3)
    
    # Volume spike: 2x 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        s3_val = s3_aligned[i]
        r3_val = r3_aligned[i]
        
        if position == 0:
            # Long: break above R3 (strong resistance) with volume spike
            if price > r3_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 (strong support) with volume spike
            elif price < s3_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to weekly pivot or opposite extreme
            if price <= pivot[i] or price >= s3_val:  # pivot or S3 as target
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to weekly pivot or opposite extreme
            if price >= pivot[i] or price <= r3_val:  # pivot or R3 as target
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_WeeklyPivot_S3R3_Breakout_Volume"
timeframe = "12h"
leverage = 1.0