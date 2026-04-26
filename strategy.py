#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session_v1
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume spike confirmation, filtered to 08-20 UTC session.
- Uses 1h timeframe for precise entry timing with 4h/1d for signal direction
- Camarilla R1/S1 levels from 1d provide precise support/resistance from prior day
- 4h EMA34 filter ensures trades align with higher timeframe trend
- Volume spike (2.0x 20-period average) confirms institutional participation
- Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
- Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the 4h trend and using volume to filter false breakouts
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
    
    # Pre-compute session hours for efficiency
    hours = prices.index.hour
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + (1.0/12) * (high_1d - low_1d)
    camarilla_s1 = close_1d - (1.0/12) * (high_1d - low_1d)
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 4h EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: flatten position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r1_aligned[i]
        breakout_short = close[i] < camarilla_s1_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 AND close > 4h EMA34 AND volume spike
            if breakout_long and close[i] > ema34_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: breakout below S1 AND close < 4h EMA34 AND volume spike
            elif breakout_short and close[i] < ema34_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: breakout below S1 (reversal signal)
            if breakout_short:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: breakout above R1 (reversal signal)
            if breakout_long:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0