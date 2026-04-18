#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Donchian channels breakouts capture momentum moves.
# 12h EMA34 ensures we trade only in the direction of higher timeframe trend.
# Volume confirmation adds conviction to breakouts.
# Designed for low trade frequency (20-50/year) to minimize fee drag.
# Works in bull markets (breakouts above upper band with 12h uptrend) and bear markets (breakouts below lower band with 12h downtrend).

name = "4h_Donchian20_12hEMA34_Volume_Filter"
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
    
    # Get 12h data for trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian channels (20-period) using previous period's data to avoid look-ahead
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    upper_band = high_20
    lower_band = low_20
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above upper band AND 12h EMA34 above price (uptrend) AND volume confirmation
            long_breakout = close[i] > upper_band[i]
            uptrend = ema_34_12h_aligned[i] > close[i]
            if vol_confirm and uptrend and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND 12h EMA34 below price (downtrend) AND volume confirmation
            elif vol_confirm and (ema_34_12h_aligned[i] < close[i]) and close[i] < lower_band[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower band OR 12h EMA34 crosses below price (trend change)
            exit_condition = close[i] < lower_band[i] or ema_34_12h_aligned[i] < close[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper band OR 12h EMA34 crosses above price (trend change)
            exit_condition = close[i] > upper_band[i] or ema_34_12h_aligned[i] > close[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals