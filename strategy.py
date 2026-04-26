#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrendFilter_VolumeConfirm_v1
Hypothesis: 12h Camarilla pivot breakout strategy with 1d trend filter and volume confirmation.
- Uses 12h timeframe for low trade frequency (target: 12-37 trades/year)
- Camarilla pivot levels (R3, S3) from 1d data act as strong support/resistance
- 1d EMA34 filter ensures trades align with daily trend (bull/bear agnostic)
- Volume spike (>1.5x 20-period average) confirms breakout momentum
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the daily trend and using Camarilla levels for entry
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (R3, S3) from 1d OHLC
    # Camarilla formulas:
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume spike filter (20-period average on 12h data)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA and 20 for volume MA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions
        breakout_above_r3 = close[i] > camarilla_r3_aligned[i]
        breakout_below_s3 = close[i] < camarilla_s3_aligned[i]
        
        # 1d trend filter
        daily_uptrend = close[i] > ema34_1d_aligned[i]
        daily_downtrend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: breakout above R3 AND daily uptrend AND volume spike
            if breakout_above_r3 and daily_uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND daily downtrend AND volume spike
            elif breakout_below_s3 and daily_downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls back below R3 OR daily trend turns down
            if close[i] < camarilla_r3_aligned[i] or not daily_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises back above S3 OR daily trend turns up
            if close[i] > camarilla_s3_aligned[i] or not daily_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrendFilter_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0