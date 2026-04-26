#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session
Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter and 1h volume spike confirmation.
- Uses 1h timeframe for precise entry timing while using 4h for signal direction (proven to reduce fee drag)
- Camarilla R1/S1 levels from 4h provide strong support/resistance from prior 4h candle
- 4h EMA20 filter ensures trades align with higher timeframe trend
- 1h volume spike (>2x 20-period average) confirms institutional participation
- Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
- Designed for 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the 4h trend and using volume spike to filter false breakouts
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
    
    # Pre-compute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate 1h volume spike confirmation (>2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # Calculate Camarilla levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    camarilla_r1 = close_4h + (1.0/6) * (high_4h - low_4h)  # R1 = close + 1/6*(high-low)
    camarilla_s1 = close_4h - (1.0/6) * (high_4h - low_4h)  # S1 = close - 1/6*(high-low)
    
    # Align Camarilla levels to 1h timeframe (use previous 4h bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for 4h EMA, 20 for volume MA)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r1_aligned[i]
        breakout_short = close[i] < camarilla_s1_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 AND close > 4h EMA20 AND volume spike
            if breakout_long and close[i] > ema20_4h_aligned[i] and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: breakout below S1 AND close < 4h EMA20 AND volume spike
            elif breakout_short and close[i] < ema20_4h_aligned[i] and vol_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: breakout below S1 OR close < 4h EMA20
            if breakout_short or close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: breakout above R1 OR close > 4h EMA20
            if breakout_long or close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0