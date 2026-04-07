#!/usr/bin/env python3
"""
6h_weekly_pivot_reversal_v1
Hypothesis: On 6-hour timeframe, use weekly pivot points (R3/S3) for mean reversion signals.
When price reaches weekly R3 or S3 with rejection candle (pin bar) and volume confirmation,
enter opposite direction. Weekly pivot acts as strong support/resistance in both bull and bear markets.
Exit when price returns to weekly pivot point or opposite S1/R1 level.
Targets 15-25 trades/year to minimize fee drag while capturing high-probability reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L
    # S2 = P-(H-L), R2 = P+(H-L), S3 = L-2*(H-P), R3 = H+2*(P-L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    s1 = 2 * pp - weekly_high
    r1 = 2 * pp - weekly_low
    s2 = pp - (weekly_high - weekly_low)
    r2 = pp + (weekly_high - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    r3 = weekly_high + 2 * (pp - weekly_low)
    
    # Align to 6h timeframe (shifted by 1 week to avoid look-ahead)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # Volume confirmation (24-period average = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):  # Start after volume MA warmup
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 24-period average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Pin bar detection: small body, long wick in direction of rejection
        body_size = abs(close[i] - open_price[i])
        total_range = high[i] - low[i]
        lower_wick = min(open_price[i], close[i]) - low[i]
        upper_wick = high[i] - max(open_price[i], close[i])
        
        # Avoid division by zero
        if total_range == 0:
            signals[i] = 0.0
            continue
            
        # Bullish pin bar: long lower wick, small body
        bullish_pin = (lower_wick > 0.6 * total_range) and (body_size < 0.3 * total_range)
        # Bearish pin bar: long upper wick, small body
        bearish_pin = (upper_wick > 0.6 * total_range) and (body_size < 0.3 * total_range)
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price returns to weekly pivot or reaches S1
            if close[i] >= pp_aligned[i] or close[i] <= s1_aligned[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price returns to weekly pivot or reaches R1
            if close[i] <= pp_aligned[i] or close[i] >= r1_aligned[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price at or below S3 with bullish pin bar and volume confirmation
            long_entry = (low[i] <= s3_aligned[i]) and bullish_pin and vol_confirm
            
            # Short entry: price at or above R3 with bearish pin bar and volume confirmation
            short_entry = (high[i] >= r3_aligned[i]) and bearish_pin and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals