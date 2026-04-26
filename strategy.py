#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: 4h Camarilla pivot R1/S1 breakout with 12-hour EMA34 trend filter and volume confirmation.
Only trade in direction of 12h trend: long R1 breakout in uptrend, short S1 breakdown in downtrend.
Volume spike (>1.5x 20-period average) confirms breakout strength.
Designed for 19-50 trades/year (75-200 over 4 years) by requiring confluence of breakout, trend, and volume.
Works in bull/bear via 12h trend filter: only takes long breakouts in uptrend, short in downtrend.
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Load 12h data ONCE before loop for HTF trend and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA34 for HTF trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    htf_trend = np.where(close > ema_34_12h_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Camarilla pivot levels from 12h data
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    R1_12h = typical_price_12h + (1.1/12) * (df_12h['high'] - df_12h['low'])  # R1 level
    S1_12h = typical_price_12h - (1.1/12) * (df_12h['high'] - df_12h['low'])  # S1 level
    
    # Align Camarilla levels to 4h timeframe
    R1_12h_aligned = align_htf_to_ltf(prices, df_12h, R1_12h.values)
    S1_12h_aligned = align_htf_to_ltf(prices, df_12h, S1_12h.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 12h EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R1_12h_aligned[i]) or np.isnan(S1_12h_aligned[i])):
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
        
        # Breakout conditions with trend filter
        if htf_trend[i] == 1:  # Uptrend on 12h
            # Long breakout above R1 with volume spike
            if close[i] > R1_12h_aligned[i] and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if price falls below S1 (reversal signal)
            elif position == 1 and close[i] < S1_12h_aligned[i]:
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
            if close[i] < S1_12h_aligned[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if price rises above R1 (reversal signal)
            elif position == -1 and close[i] > R1_12h_aligned[i]:
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