#!/usr/bin/env python3
# 12h_camarilla_pivot_weekly_trend_volume_v1
# Hypothesis: Camarilla pivot levels from 1-day combined with weekly trend filter and volume confirmation.
# In uptrend (price > weekly EMA20), buy at S3 level with volume spike; in downtrend (price < weekly EMA20), sell at R3 level with volume spike.
# Uses mean-reversion at extreme pivot levels within the prevailing weekly trend.
# Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_weekly_trend_volume_v1"
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
    
    # Weekly trend filter (EMA20 on weekly close) - load once before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on weekly data
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.2500)
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.2500)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    
    # Volume confirmation (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema20_1w_aligned[i]
        weekly_downtrend = close[i] < ema20_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: price reaches R3 or trend changes
            if close[i] >= camarilla_r3_aligned[i] or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price reaches S3 or trend changes
            if close[i] <= camarilla_s3_aligned[i] or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long entry: price at S3 level in uptrend
                if weekly_uptrend and close[i] <= camarilla_s3_aligned[i] * 1.001 and close[i] >= camarilla_s3_aligned[i] * 0.999:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price at R3 level in downtrend
                elif weekly_downtrend and close[i] >= camarilla_r3_aligned[i] * 0.999 and close[i] <= camarilla_r3_aligned[i] * 1.001:
                    position = -1
                    signals[i] = -0.25
    
    return signals