#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d EMA34 is rising AND volume > 1.5x 20-period MA.
# Short when price breaks below Donchian(20) low AND 1d EMA34 is falling AND volume > 1.5x 20-period MA.
# Exit when price crosses 12h EMA10 (trend reversal signal) OR volume drops below average.
# Uses 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Donchian channels provide clear trend structure, EMA34 filters for higher-timeframe trend direction,
# volume confirmation ensures institutional participation. Designed for BTC/ETH resilience in both bull/bear regimes.

name = "12h_Donchian20_1dEMA34_VolumeSpike_TrendFilter"
timeframe = "12h"
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
    
    # Session filter: 08-20 UTC (avoid low-liquidity hours)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Donchian(20) - highest high/lowest low of past 20 bars
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA10 for exit signal
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 12h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_10[i]) or 
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 12h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 1.5)
        
        # 1d EMA34 trend direction (rising/falling)
        ema_34_rising = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
        ema_34_falling = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
        
        if position == 0:
            # Long: price breaks above Donchian high AND EMA34 rising AND volume spike AND session
            if close[i] > highest_high_20[i] and ema_34_rising and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND EMA34 falling AND volume spike AND session
            elif close[i] < lowest_low_20[i] and ema_34_falling and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA10 OR volume drops below average
            if close[i] < ema_10[i] or volume[i] < volume_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA10 OR volume drops below average
            if close[i] > ema_10[i] or volume[i] < volume_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals