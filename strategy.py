#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian channel breakout with volume confirmation and 1d EMA trend filter.
# Enter long when price breaks above 12h Donchian upper band with volume spike and price > 1d EMA34 (bullish trend).
# Enter short when price breaks below 12h Donchian lower band with volume spike and price < 1d EMA34 (bearish trend).
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 12-37 trades/year.
# Donchian provides structure from higher timeframe, volume confirms breakout strength, EMA filter ensures trend alignment.
# Works in bull (breakouts with trend) and bear (failed breaks reverse via exits) markets.

name = "6h_Donchian20_12hBreakout_Volume_EMA34Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channel (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    n_12h = len(high_12h)
    donchian_upper = np.full(n_12h, np.nan)
    donchian_lower = np.full(n_12h, np.nan)
    
    for i in range(20, n_12h):
        donchian_upper[i] = np.max(high_12h[i-20:i])
        donchian_lower[i] = np.min(low_12h[i-20:i])
    
    # Forward fill Donchian levels
    donchian_upper = pd.Series(donchian_upper).ffill().values
    donchian_lower = pd.Series(donchian_lower).ffill().values
    
    # Align 12h indicators to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions with volume confirmation and EMA trend filter
        long_breakout = close[i] > donchian_upper_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]
        
        # Exit conditions: opposite Donchian level
        long_exit = close[i] < donchian_lower_aligned[i]
        short_exit = close[i] > donchian_upper_aligned[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
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