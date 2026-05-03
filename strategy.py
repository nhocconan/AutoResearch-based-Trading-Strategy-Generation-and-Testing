#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter, volume confirmation, and ATR-based stoploss.
# Long when price breaks above Donchian(20) high AND 12h EMA50 is rising AND volume > 1.5x 20-period MA.
# Short when price breaks below Donchian(20) low AND 12h EMA50 is falling AND volume > 1.5x 20-period MA.
# Exit via ATR trailing stop: long exits when price < highest high since entry - 2.5*ATR, short exits when price > lowest low since entry + 2.5*ATR.
# Uses 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with strict entry conditions.
# Donchian channels provide clear breakout levels, EMA50 filters for trend direction, volume confirms participation.
# Designed to work in both bull (breakouts with trend) and bear (breakdowns with trend) markets.

name = "4h_Donchian20_12hEMA50_VolumeSpike_ATRStop"
timeframe = "4h"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.zeros(n)  # highest high since long entry
    lowest_since_entry = np.zeros(n)   # lowest low since short entry
    
    for i in range(20, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            # Carry forward stop levels
            if i > 0:
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            continue
        
        # Update stop levels if in position
        if position == 1:
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
        elif position == -1:
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
        else:
            # Reset stop levels when flat
            highest_since_entry[i] = highest_high[i]
            lowest_since_entry[i] = lowest_low[i]
        
        # Volume spike condition: current 4h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 1.5)
        
        # 12h EMA50 trend: rising if current > previous, falling if current < previous
        ema_rising = i > 0 and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
        ema_falling = i > 0 and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]
        
        if position == 0:
            # Long: price breaks above Donchian high AND EMA50 rising AND volume spike AND session
            if close[i] > highest_high[i] and ema_rising and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]
            # Short: price breaks below Donchian low AND EMA50 falling AND volume spike AND session
            elif close[i] < lowest_low[i] and ema_falling and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]
        elif position == 1:
            # Long exit: price < highest high since entry - 2.5*ATR
            if close[i] < highest_since_entry[i] - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > lowest low since entry + 2.5*ATR
            if close[i] > lowest_since_entry[i] + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals