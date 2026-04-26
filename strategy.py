#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_v1
Hypothesis: 1h Camarilla pivot R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
Only trade breakouts in direction of 4h trend (EMA50) to avoid counter-trend whipsaws.
Volume spike (>1.5x 20-period average) confirms institutional participation.
Discrete position sizing (0.20) minimizes fee churn. Session filter (08-20 UTC) reduces noise.
Designed for 1h timeframe targeting 15-37 trades/year (60-150 over 4 years) by requiring confluence.
Works in bull/bear via 4h trend filter: in uptrend, favor longs; in downtrend, favor shorts.
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
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed)
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h.values)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 4h EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or
            not in_session[i]):
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
            # Mean reversion short: breakdown below S1 in uptrend (fade the move)
            elif close[i] < S1_4h_aligned[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = -0.20
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
            # Mean reversion long: breakout above R1 in downtrend (fade the move)
            elif close[i] > R1_4h_aligned[i] and volume_spike:
                if position != 1:
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.20
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

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0