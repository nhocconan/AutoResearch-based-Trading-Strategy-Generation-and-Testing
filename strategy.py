#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian channel breakout with 1d EMA34 trend filter and volume confirmation.
# Enter long when price breaks above 12h Donchian upper (20) with 1d EMA34 uptrend and volume > 1.5x 20-bar avg.
# Enter short when price breaks below 12h Donchian lower (20) with 1d EMA34 downtrend and volume > 1.5x 20-bar avg.
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 12-30 trades/year.
# Donchian provides structure from higher timeframe, EMA34 filters trend direction, volume confirms breakout strength.
# Works in bull (breakouts with trend) and bear (failed breaks via exits) markets.

name = "6h_Donchian20_12hBreakout_1dEMA34_VolumeFilter_v1"
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
    
    # Calculate 12h Donchian channel (20)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    n_12h = len(high_12h)
    donchian_upper = np.full(n_12h, np.nan)
    donchian_lower = np.full(n_12h, np.nan)
    
    for i in range(20, n_12h):
        # Use rolling window of 20 completed bars
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
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 1d EMA34 trend: slope > 0 for uptrend, < 0 for downtrend
    ema_34_slope = np.diff(ema_34_aligned, prepend=ema_34_aligned[0])
    ema_uptrend = ema_34_slope > 0
    ema_downtrend = ema_34_slope < 0
    
    # Calculate 6h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions with trend filter and volume confirmation
        long_breakout = close[i] > donchian_upper_aligned[i] and ema_uptrend[i] and volume_confirm[i]
        short_breakout = close[i] < donchian_lower_aligned[i] and ema_downtrend[i] and volume_confirm[i]
        
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