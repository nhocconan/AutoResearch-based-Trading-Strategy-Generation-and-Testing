#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses Donchian channel from previous 20 periods for robust price structure
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend entries
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Targets 19-50 trades/year (75-200 total over 4 years) for 4h timeframe
# Works in bull markets via upper channel breakout continuation and in bear markets via lower channel breakdown continuation

name = "4h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channel (20) from previous periods
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    # Using previous bar to avoid look-ahead
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    
    # Rolling window of 20 on shifted arrays
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high_shift[i-20:i])
        lower[i] = np.min(low_shift[i-20:i])
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and volume MA)
    start_idx = 40  # buffer for 20-period calculations
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper + 1d close > EMA50 + volume spike
            if (close[i] > upper[i] and 
                close[i] > ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + 1d close < EMA50 + volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price drops below Donchian lower or 1d trend breaks
            if close[i] < lower[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper or 1d trend breaks
            if close[i] > upper[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals