#!/usr/bin/env python3
# 6h_daily_camarilla_pivot_reversal_v1
# Hypothesis: Camarilla pivot levels from 1-day timeframe act as strong intraday support/resistance on 6h chart.
# Fade trades at R3/S3 levels with volume confirmation, breakout continuation at R4/S4 with volume spike.
# Works in both bull and bear markets as Camarilla levels adapt to recent volatility.
# Primary timeframe: 6h, HTF: 1d for pivot calculation. Target: 12-35 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_daily_camarilla_pivot_reversal_v1"
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
    
    # Calculate Camarilla pivots from 1d timeframe (updated once per day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's OHLC
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6), R1 = C + ((H-L)*1.1/12)
    # S1 = C - ((H-L)*1.1/12), S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # where C = (H+L+C)/3 (typical price), but Camarilla uses close of previous day as pivot
    # Actually standard Camarilla: Pivot = (H+L+C)/3
    # But many versions use close as base. Let's use standard: pivot = (H+L+C)/3
    
    # We need previous day's OHLC for today's levels
    # So we shift the 1d data by 1 to avoid look-ahead
    df_1d_prev = df_1d.shift(1)
    
    typical_price = (df_1d_prev['high'].values + df_1d_prev['low'].values + df_1d_prev['close'].values) / 3
    # Actually standard Camarilla pivot is (H+L+C)/3, but some use (H+L+2C)/4 or just C
    # Let's use the most common: pivot = (H+L+C)/3
    pivot = typical_price
    high_low = df_1d_prev['high'].values - df_1d_prev['low'].values
    
    # Calculate levels
    R4 = pivot + (high_low * 1.1 / 2)
    R3 = pivot + (high_low * 1.1 / 4)
    R2 = pivot + (high_low * 1.1 / 6)
    R1 = pivot + (high_low * 1.1 / 12)
    S1 = pivot - (high_low * 1.1 / 12)
    S2 = pivot - (high_low * 1.1 / 6)
    S3 = pivot - (high_low * 1.1 / 4)
    S4 = pivot - (high_low * 1.1 / 2)
    
    # Align all levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d_prev, pivot)
    R1_6h = align_htf_to_ltf(prices, df_1d_prev, R1)
    R2_6h = align_htf_to_ltf(prices, df_1d_prev, R2)
    R3_6h = align_htf_to_ltf(prices, df_1d_prev, R3)
    R4_6h = align_htf_to_ltf(prices, df_1d_prev, R4)
    S1_6h = align_htf_to_ltf(prices, df_1d_prev, S1)
    S2_6h = align_htf_to_ltf(prices, df_1d_prev, S2)
    S3_6h = align_htf_to_ltf(prices, df_1d_prev, S3)
    S4_6h = align_htf_to_ltf(prices, df_1d_prev, S4)
    
    # Volume confirmation: 6h volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any pivot levels or volume data is NaN
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(R4_6h[i]) or 
            np.isnan(S4_6h[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                # Hold current position until exit signal
                signals[i] = 0.30 if position == 1 else -0.30
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_ratio[i] > 1.5  # Volume confirmation
        
        if position == 1:  # Long position
            # Exit conditions: price breaks below S3 (stop) or reaches R1 (target)
            if price < S3_6h[i] or price > R1_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit conditions: price breaks above R3 (stop) or reaches S1 (target)
            if price > R3_6h[i] or price < S1_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat - look for entry
            # Fade at R3/S3 with volume confirmation
            if vol_ok:
                if price > R3_6h[i] and price < R4_6h[i]:
                    # Potential short at R3 resistance
                    position = -1
                    signals[i] = -0.30
                elif price < S3_6h[i] and price > S4_6h[i]:
                    # Potential long at S3 support
                    position = 1
                    signals[i] = 0.30
            # Breakout continuation at R4/S4 with volume spike
            if vol_ok and vol_ratio[i] > 2.0:  # Strong volume spike
                if price > R4_6h[i]:
                    # Breakout above R4 - go long
                    position = 1
                    signals[i] = 0.30
                elif price < S4_6h[i]:
                    # Breakdown below S4 - go short
                    position = -1
                    signals[i] = -0.30
    
    return signals