#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability reversal/breakout zones. 1w EMA34 ensures alignment with higher timeframe trend.
# Volume spike confirms institutional participation. Designed for 20-50 trades/year on 4h to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via breakdown shorts.

name = "4h_Camarilla_R3S3_1wEMA34_VolumeSpike"
timeframe = "4h"
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Need at least 2 bars for Camarilla calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla pivot levels (based on previous bar)
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_range = prev_high - prev_low
        
        # Calculate Camarilla levels
        R3 = prev_close + (prev_range * 1.1 / 2)
        S3 = prev_close - (prev_range * 1.1 / 2)
        
        # Volume confirmation: 20-period EMA on 4h
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (2.0 * vol_ema_20)  # Higher threshold for fewer trades
        
        if position == 0:
            # Long: price breaks above Camarilla R3 in 1w uptrend with volume spike
            if close[i] > R3 and ema_34_1w_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 in 1w downtrend with volume spike
            elif close[i] < S3 and ema_34_1w_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla R3 or loses 1w uptrend
            if close[i] < R3 or ema_34_1w_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla S3 or loses 1w downtrend
            if close[i] > S3 or ema_34_1w_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals