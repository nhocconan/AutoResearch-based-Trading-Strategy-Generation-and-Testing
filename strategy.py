#!/usr/bin/env python3
name = "4h_4hDonchianBreakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-period high with volume spike in 1d uptrend
            if close[i] > high_20[i] and volume[i] > vol_ma_20[i] * 2.0 and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with volume spike in 1d downtrend
            elif close[i] < low_20[i] and volume[i] > vol_ma_20[i] * 2.0 and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to 20-period low (trailing stop logic)
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to 20-period high
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume spike confirmation
# - Uses 4h Donchian channels (20-period high/low) for breakout detection
# - 1d EMA34 trend filter ensures trades align with daily trend (uptrend for long, downtrend for short)
# - Volume spike (>2x 20-period average) confirms breakout validity
# - Exits when price returns to the opposite Donchian band (natural mean reversion within the channel)
# - Works in both bull and bear markets due to trend filter and bidirectional breakout logic
# - Position size 0.25 balances risk and return while minimizing fee churn
# - Target: 20-50 trades per year (80-200 over 4 years) to stay within optimal trade frequency range
# - Donchian breakouts capture momentum while trend filter reduces whipsaws
# - Volume confirmation reduces false breakouts from low-volume spikes
# - Simple, robust logic with clear entry/exit conditions (no complex calculations)
# - Avoids overtrading by requiring multiple confluence factors (breakout + trend + volume)