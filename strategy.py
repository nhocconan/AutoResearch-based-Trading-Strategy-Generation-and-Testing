#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h EMA50 trend filter and volume spike confirmation
# Elder Ray measures bull/bear strength relative to EMA13. In strong trends, one power dominates while the other fades.
# Enter long when Bull Power > 0, Bear Power < 0, price > 12h EMA50, and volume spike.
# Enter short when Bear Power < 0, Bull Power > 0, price < 12h EMA50, and volume spike.
# Uses discrete position sizing (0.25) to limit fee drag. Target: 50-150 total trades over 4 years.
# Works in bull markets via sustained Bull Power > 0 and in bear markets via sustained Bear Power < 0.

name = "6h_ElderRay_12hEMA50_VolumeSpike"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Elder Ray on 6h data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 20-period EMA on 6h
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start from 13 to have valid EMA13
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price above 12h EMA50, volume spike
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power > 0, price below 12h EMA50, volume spike
            elif bear_power[i] < 0 and bull_power[i] > 0 and close[i] < ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or Bear Power >= 0 or price below 12h EMA50
            if bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or Bull Power <= 0 or price above 12h EMA50
            if bear_power[i] >= 0 or bull_power[i] <= 0 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals