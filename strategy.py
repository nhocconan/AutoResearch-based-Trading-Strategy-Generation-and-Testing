#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation
# Uses Donchian channel from daily timeframe for structural breakouts
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend entries
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Targets 7-25 trades/year (30-100 total over 4 years) for 1d timeframe
# Works in bull markets via upper channel breakout continuation and in bear markets via lower channel breakdown continuation

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Donchian channel (20-period) from previous daily bar
    # Using previous bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan  # First bar has no previous
    prev_low[0] = np.nan
    
    # Calculate 20-period rolling max/min on previous values
    donchian_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and volume calculations)
    start_idx = 20  # buffer for 20-period calculations
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high + 1w close > EMA50 + volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + 1w close < EMA50 + volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price drops below Donchian low or 1w trend breaks
            if close[i] < donchian_low[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high or 1w trend breaks
            if close[i] > donchian_high[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals