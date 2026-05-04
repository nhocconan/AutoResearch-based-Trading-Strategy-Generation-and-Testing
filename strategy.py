#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (jaw=13, teeth=8, lips=5) identifies trends via SMAs with forward shift
# Trend filter: 1d EMA50 ensures alignment with higher timeframe direction
# Volume confirmation: >1.3x 20-period volume EMA filters low-participation signals
# Session filter: 08-20 UTC to avoid low liquidity periods
# Discrete sizing 0.25 balances risk and minimizes fee churn
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Works in bull/bear: Alligator identifies trends, EMA50 filter reduces whipsaw

name = "4h_WilliamsAlligator_1dEMA50_VolumeConfirm_Session"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator components on 4h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(high + low).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(high + low).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(high + low).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) + uptrend + volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema50_aligned[i] and volume[i] > (1.3 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) + downtrend + volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema50_aligned[i] and volume[i] > (1.3 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR trend changes OR volume drops
            if (lips[i] <= teeth[i] or 
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR trend changes OR volume drops
            if (lips[i] >= teeth[i] or 
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals