#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 6h with 1d EMA34 trend filter and volume confirmation. Designed for 12-25 trades/year on BTC/ETH/SOL. Uses wider R3/S3 levels for fewer, higher-quality breakouts. 1d EMA34 provides longer-term trend filter to avoid counter-trend trades. Volume spike confirms breakout legitimacy. Should work in bull (trend-aligned breakouts) and bear (avoids false breakouts via trend filter).
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    R3 = PP + (high_1d - low_1d) * 1.5 / 2.0  # R3 = PP + 1.5*(H-L)/2
    S3 = PP - (high_1d - low_1d) * 1.5 / 2.0  # S3 = PP - 1.5*(H-L)/2
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(100, 34, 20)  # EMA, volume avg
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_1d_aligned[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout in direction of 1d trend with volume confirmation
            # Long: price above 1d EMA34 AND break above R3 + volume spike
            long_entry = (close_val > ema_trend) and (close_val > R3_aligned[i]) and volume_spike[i]
            # Short: price below 1d EMA34 AND break below S3 + volume spike
            short_entry = (close_val < ema_trend) and (close_val < S3_aligned[i]) and volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on S3 retracement
            if close_val < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R3 retracement
            if close_val > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0