#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla range
    range_1d = high_1d - low_1d
    
    # Camarilla levels (based on previous day)
    # R4 = close + 1.5 * range
    # R3 = close + 1.1 * range
    # R2 = close + 0.6 * range
    # R1 = close + 0.3 * range
    # S1 = close - 0.3 * range
    # S2 = close - 0.6 * range
    # S3 = close - 1.1 * range
    # S4 = close - 1.5 * range
    
    r1 = close_1d + 0.3 * range_1d
    r2 = close_1d + 0.6 * range_1d
    r3 = close_1d + 1.1 * range_1d
    r4 = close_1d + 1.5 * range_1d
    s1 = close_1d - 0.3 * range_1d
    s2 = close_1d - 0.6 * range_1d
    s3 = close_1d - 1.1 * range_1d
    s4 = close_1d - 1.5 * range_1d
    
    # Align to 6h timeframe (previous day's levels available at next 6h bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume / 20-period average volume
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long at S3/S4 bounce or break above R4
            if vol_ratio_val > 1.3:  # Volume confirmation
                # Bounce from S3 (strong support)
                if price_close > s3_aligned[i] and price_close < s2_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Breakout above R4 (strong bullish)
                elif price_close > r4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bounce from S4 (extreme support)
                elif price_close > s4_aligned[i] and price_close < s3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short at R3/R4 rejection or break below S4
            elif vol_ratio_val > 1.3:  # Volume confirmation
                # Rejection at R3 (strong resistance)
                if price_close < r3_aligned[i] and price_close > r2_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                # Breakdown below S4 (strong bearish)
                elif price_close < s4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                # Rejection at R4 (extreme resistance)
                elif price_close < r4_aligned[i] and price_close > r3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:  # Long position
                # Exit if price reaches R1 (first resistance) or breaks below S1
                if price_close >= r1_aligned[i] or price_close <= s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                # Hold
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Exit if price reaches S1 (first support) or breaks above R1
                if price_close <= s1_aligned[i] or price_close >= r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                # Hold
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_S3_S4_R3_R4_Bounce_Breakout"
timeframe = "6h"
leverage = 1.0