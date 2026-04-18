#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Uses 12h EMA34 as trend filter to avoid counter-trend trades.
# Volume confirmation ensures breakouts have conviction.
# Designed for low trade frequency (15-30/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (long breakouts above upper band with 12h uptrend) and bear markets (short breakouts below lower band with 12h downtrend).
name = "4h_Donchian20_12hEMA34_Volume"
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
            # Long: price breaks above upper band AND 12h uptrend AND volume confirmation
            long_breakout = close[i] > upper_band[i]
            uptrend = close_12h[-1] > ema_34_12h[-1] if len(close_12h) > 0 else False  # Simplified trend check
            if vol_confirm and long_breakout and ema_34_12h_aligned[i] > 0:  # Using EMA value as trend proxy
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND 12h downtrend AND volume confirmation
            elif vol_confirm and close[i] < lower_band[i] and ema_34_12h_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower band
            if close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper band
            if close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals