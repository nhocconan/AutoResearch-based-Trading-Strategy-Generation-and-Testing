#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. 
# Long: Bear Power < 0 AND Bull Power rising (momentum) in 1d uptrend with volume confirmation
# Short: Bull Power > 0 AND Bear Power falling (momentum) in 1d downtrend with volume confirmation
# Works in both bull and bear markets by capturing momentum shifts in the direction of higher timeframe trend.
# Target: 12-37 trades/year on 6h to minimize fee drag.

name = "6h_ElderRay_Momentum_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate momentum of Elder Ray components (1-period change)
    bull_power_momentum = np.diff(bull_power, prepend=bull_power[0])
    bear_power_momentum = np.diff(bear_power, prepend=bear_power[0])
    
    # Volume confirmation: 20-period EMA
    volume_s = pd.Series(volume)
    vol_ema_20 = volume_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after sufficient warmup for EMA13
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bear Power negative (bears weak) AND Bull Power rising (momentum) in 1d uptrend with volume spike
            if (bear_power[i] < 0 and bull_power_momentum[i] > 0 and 
                ema_34_1d_aligned[i] < close[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power positive (bulls weak) AND Bear Power falling (momentum) in 1d downtrend with volume spike
            elif (bull_power[i] > 0 and bear_power_momentum[i] < 0 and 
                  ema_34_1d_aligned[i] > close[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power becomes positive OR loses 1d uptrend
            if bear_power[i] >= 0 or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power becomes negative OR loses 1d downtrend
            if bull_power[i] <= 0 or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals