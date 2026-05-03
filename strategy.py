#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1w EMA50 is rising AND volume > 1.5x 20-period MA.
# Short when price breaks below Donchian(20) low AND 1w EMA50 is falling AND volume > 1.5x 20-period MA.
# Exit when price crosses Donchian(20) midpoint OR 1w EMA50 flips direction.
# Uses 1d timeframe to achieve 30-100 total trades over 4 years (7-25/year) with strict entry conditions.
# Donchian provides clear structure, 1w EMA50 filters for strong weekly trend, volume confirms participation.
# Designed to work in bull markets (breakouts with rising EMA) and bear markets (breakdowns with falling EMA).

name = "1d_Donchian20_1wEMA50_VolumeSpike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian(20)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (highest_20 + lowest_20) / 2.0
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current 1d volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 1.5)
        
        # 1w EMA50 direction: rising if current > previous, falling if current < previous
        ema_rising = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
        ema_falling = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        
        if position == 0:
            # Long: break above Donchian high AND EMA rising AND volume spike AND session
            if close[i] > highest_20[i] and ema_rising and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND EMA falling AND volume spike AND session
            elif close[i] < lowest_20[i] and ema_falling and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: cross below Donchian midpoint OR EMA starts falling
            if close[i] < midpoint_20[i] or ema_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: cross above Donchian midpoint OR EMA starts rising
            if close[i] > midpoint_20[i] or ema_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals