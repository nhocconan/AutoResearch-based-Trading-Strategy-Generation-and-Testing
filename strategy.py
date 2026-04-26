#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: 4h Camarilla pivot R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
In trending markets (price > 12h EMA50), long R1 breakout or short S1 breakout with volume > 2x 20-period average.
Only takes breakouts in direction of 12h trend to avoid counter-trend whipsaws.
Designed for 19-50 trades/year (75-200 over 4 years) by requiring confluence of breakout, trend, and volume.
Works in bull/bear via 12h trend filter: only takes long breakouts in uptrend, short in downtrend.
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Load 12h data ONCE before loop for HTF trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for HTF trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    htf_trend = np.where(close > ema_50_12h_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 20-period volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla pivot levels from 1d data
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R1_1d = typical_price_1d + (1.1/6) * (df_1d['high'] - df_1d['low'])
    S1_1d = typical_price_1d - (1.1/6) * (df_1d['high'] - df_1d['low'])
    
    # Align Camarilla levels to 4h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d.values)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 12h EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Breakout conditions with trend filter
        if htf_trend[i] == 1:  # Uptrend on 12h
            # Long breakout above R1 with volume spike
            if close[i] > R1_1d_aligned[i] and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if price falls below S1 (reversal signal)
            elif position == 1 and close[i] < S1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 12h
            # Short breakdown below S1 with volume spike
            if close[i] < S1_1d_aligned[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if price rises above R1 (reversal signal)
            elif position == -1 and close[i] > R1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
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

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0