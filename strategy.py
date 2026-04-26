#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_WeeklyTrend_VolumeSpike_v1
Hypothesis: 6h Camarilla pivot R3/S3 breakout with 1w trend filter and volume confirmation.
Only trade breakouts in direction of weekly EMA34 trend (price > EMA34 = uptrend, price < EMA34 = downtrend).
In uptrend: long R3 breakouts with volume spike; in downtrend: short S3 breakdowns with volume spike.
Adds 1w trend filter to avoid counter-trend trades and reduce whipsaw in bear markets.
Uses volume confirmation to avoid false breakouts and discrete position sizing (0.25) to minimize fee churn.
Target: 50-150 total trades over 4 years (12-37/year) by requiring confluence of breakout, weekly trend, and volume.
Designed for BTC/ETH - weekly trend filter provides robustness across market regimes.
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
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for HTF trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    htf_trend = np.where(close > ema_34_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d data
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R3_1d = typical_price_1d + (1.1/2) * (df_1d['high'] - df_1d['low'])  # R3 level
    S3_1d = typical_price_1d - (1.1/2) * (df_1d['high'] - df_1d['low'])  # S3 level
    
    # Align Camarilla levels to 1d timeframe (no additional delay needed)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d.values)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1w EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions with weekly trend filter
        if htf_trend[i] == 1:  # Uptrend on 1w
            # Long breakout above R3 with volume spike
            if close[i] > R3_1d_aligned[i] and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 1w
            # Short breakdown below S3 with volume spike
            if close[i] < S3_1d_aligned[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_WeeklyTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0