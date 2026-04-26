#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter_v1
Hypothesis: 1h Camarilla pivot R1/S1 breakout with 4-hour trend filter and volume confirmation.
Only trade in direction of 4h EMA50 trend: long R1 breakout in uptrend, short S1 breakdown in downtrend.
Uses volume > 1.5x 20-period average for confirmation and session filter (08-20 UTC) to reduce noise.
Designed for 15-37 trades/year (60-150 over 4 years) by requiring confluence of breakout, trend, volume, and session.
Works in bull/bear via 4h trend filter: only takes long breakouts in uptrend, short in downtrend.
Uses discrete position sizing (0.20) to minimize fee churn.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - vectorized
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for HTF trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for HTF trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    htf_trend = np.where(close > ema_50_4h_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Camarilla pivot levels from 4h data
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    R1_4h = typical_price_4h + (1.1/12) * (df_4h['high'] - df_4h['low'])  # R1 level
    S1_4h = typical_price_4h - (1.1/12) * (df_4h['high'] - df_4h['low'])  # S1 level
    
    # Align Camarilla levels to 1h timeframe
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h.values)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 4h EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            # Hold current position outside session
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions with trend filter
        if htf_trend[i] == 1:  # Uptrend on 4h
            # Long breakout above R1 with volume spike
            if close[i] > R1_4h_aligned[i] and volume_spike:
                if position != 1:
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.20
            # Exit long if price falls below S1 (reversal signal)
            elif position == 1 and close[i] < S1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
        elif htf_trend[i] == -1:  # Downtrend on 4h
            # Short breakdown below S1 with volume spike
            if close[i] < S1_4h_aligned[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = -0.20
            # Exit short if price rises above R1 (reversal signal)
            elif position == -1 and close[i] > R1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0