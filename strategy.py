#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Enter long when price breaks above 20-period 12h Donchian high, price > 1d EMA50, and volume > 1.5x 20-bar average.
# Enter short when price breaks below 20-period 12h Donchian low, price < 1d EMA50, and volume > 1.5x 20-bar average.
# Exit when price crosses back below/above the Donchian midpoint or opposite breakout occurs.
# Donchian provides clear price channels, EMA50 filters trend direction, volume confirms breakout strength.
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend with volume).
# Uses discrete position sizing (0.25) to control risk. Target: 50-150 total trades over 4 years.

name = "12h_Donchian20_1dEMA50_VolumeBreakout_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian(20) channels
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (period20_high + period20_low) / 2.0
    
    # Calculate 12h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(period20_high[i]) or 
            np.isnan(period20_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > period20_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < period20_low[i-1]  # Break below previous period's low
        
        # EMA50 trend filter
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and price_above_ema and volume_confirm[i]
        short_entry = breakout_down and price_below_ema and volume_confirm[i]
        
        # Exit conditions: price crosses Donchian midpoint or opposite breakout
        long_exit = close[i] < donchian_mid[i] or breakout_down
        short_exit = close[i] > donchian_mid[i] or breakout_up
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals