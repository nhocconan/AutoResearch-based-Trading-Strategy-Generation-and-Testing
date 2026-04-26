#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: 6h Donchian(20) breakout with 1-week trend filter and volume spike confirmation.
In strongly trending markets (price > 1-week EMA50), long upper band breakout or short lower band breakout with volume > 2x 20-period average.
Uses 1-week trend filter to ensure we only trade in the direction of the primary trend.
Designed for 12-37 trades/year (50-150 over 4 years) by requiring confluence of strong breakout, 1-week trend, and volume.
Works in bull/bear via 1-week trend filter: only takes long breakouts in uptrend, short in downtrend.
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
    
    # Load 1w data ONCE before loop for HTF trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for HTF trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    htf_trend = np.where(close > ema_50_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 20-period volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1w EMA, 20 for Donchian/volume)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(donchian_h[i]) or np.isnan(donchian_l[i])):
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
        if htf_trend[i] == 1:  # Uptrend on 1w
            # Long breakout above upper Donchian with volume spike
            if close[i] > donchian_h[i] and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if price falls below lower Donchian (reversal signal)
            elif position == 1 and close[i] < donchian_l[i]:
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
        elif htf_trend[i] == -1:  # Downtrend on 1w
            # Short breakdown below lower Donchian with volume spike
            if close[i] < donchian_l[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if price rises above upper Donchian (reversal signal)
            elif position == -1 and close[i] > donchian_h[i]:
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

name = "6h_Donchian20_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0