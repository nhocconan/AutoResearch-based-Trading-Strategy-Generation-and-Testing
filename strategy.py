#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: 4h breakout of Camarilla R3/S3 levels with 1d EMA34 trend filter and volume confirmation.
Weekly pivot levels provide strong structural support/resistance; breakouts in trend direction have higher follow-through.
Volume spike confirms institutional participation. Discrete sizing (0.30) limits fee drift.
Target: 75-200 total trades over 4 years (19-50/year) by requiring HTF alignment, breakout, trend, and volume.
Works in bull/bear: 1d EMA34 adapts to regime; volume filter avoids false breakouts in ranging markets.
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
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    camarilla_r1d = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 12
    camarilla_s1d = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 12
    camarilla_r3d = camarilla_r1d + 2 * (camarilla_r1d - camarilla_s1d)  # R3 = R1 + 4*(R1-S1)
    camarilla_s3d = camarilla_s1d - 2 * (camarilla_r1d - camarilla_s1d)  # S3 = S1 - 4*(R1-S1)
    
    # Align daily Camarilla to 4h timeframe
    camarilla_r3d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3d.values)
    camarilla_s3d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3d.values)
    
    # Calculate 4h Donchian(20) channels for entry timing
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 30 for volume MA, 20 for Donchian)
    start_idx = max(34, 30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_30[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3d_aligned[i]) or np.isnan(camarilla_s3d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 2.0 * vol_ma_30[i]
        
        # Donchian breakout conditions (use previous bar's channel to avoid look-ahead)
        breakout_above = close[i] > high_20[i-1]
        breakout_below = close[i] < low_20[i-1]
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if breakout_above and volume_spike and uptrend:
            # Long signal: breakout above Donchian high with volume, in uptrend, above daily S3
            if close[i] > camarilla_s3d_aligned[i]:
                if position != 1:
                    signals[i] = 0.30
                    position = 1
                else:
                    signals[i] = 0.30
            else:
                # Hold or flatten if not aligned with daily pivot
                if position == 1:
                    signals[i] = 0.30
                else:
                    signals[i] = 0.0
                    position = 0
        elif breakout_below and volume_spike and downtrend:
            # Short signal: breakout below Donchian low with volume, in downtrend, below daily R3
            if close[i] < camarilla_r3d_aligned[i]:
                if position != -1:
                    signals[i] = -0.30
                    position = -1
                else:
                    signals[i] = -0.30
            else:
                # Hold or flatten if not aligned with daily pivot
                if position == -1:
                    signals[i] = -0.30
                else:
                    signals[i] = 0.0
                    position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0