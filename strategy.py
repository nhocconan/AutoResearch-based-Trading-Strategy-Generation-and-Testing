#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA34 trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend via smoothed MAs.
# 1w EMA34 ensures alignment with weekly trend for higher probability trades.
# Volume spike confirms institutional participation. Designed for 10-30 trades/year on 1d.
# Works in bull markets via trend continuation and in bear markets via breakdown shorts.

name = "1d_WilliamsAlligator_1wEMA34_VolumeSpike"
timeframe = "1d"
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
    
    for i in range(13, n):  # Need at least 13 bars for Alligator Lips
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator components (13,8,5 smoothed with 8,5,3 periods respectively)
        # Jaw (13-period SMMA shifted 8 bars)
        if i >= 20:  # 13 + 8 - 1 for Jaw
            jaw_values = pd.Series(high[i-20:i+1]).rolling(window=13, min_periods=13).mean().values
            jaw = np.mean(jaw_values[-8:]) if len(jaw_values) >= 8 else np.nan
        else:
            jaw = np.nan
        
        # Teeth (8-period SMMA shifted 5 bars)
        if i >= 12:  # 8 + 5 - 1 for Teeth
            teeth_values = pd.Series(low[i-12:i+1]).rolling(window=8, min_periods=8).mean().values
            teeth = np.mean(teeth_values[-5:]) if len(teeth_values) >= 5 else np.nan
        else:
            teeth = np.nan
        
        # Lips (5-period SMMA shifted 3 bars)
        if i >= 7:  # 5 + 3 - 1 for Lips
            lips_values = pd.Series(close[i-7:i+1]).rolling(window=5, min_periods=5).mean().values
            lips = np.mean(lips_values[-3:]) if len(lips_values) >= 3 else np.nan
        else:
            lips = np.nan
        
        # Volume confirmation: 20-period EMA on 1d
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) in 1w uptrend with volume spike
            if (lips > teeth > jaw) and (ema_34_1w_aligned[i] < close[i]) and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) in 1w downtrend with volume spike
            elif (lips < teeth < jaw) and (ema_34_1w_aligned[i] > close[i]) and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks or loses 1w uptrend
            if not (lips > teeth > jaw) or (ema_34_1w_aligned[i] >= close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks or loses 1w downtrend
            if not (lips < teeth < jaw) or (ema_34_1w_aligned[i] <= close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals