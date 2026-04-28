#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h EMA34 trend filter and 1d Donchian(20) breakout with volume confirmation.
# Enter long when price breaks above 1d Donchian(20) high with volume > 2x 20-bar average and price > 4h EMA34 (uptrend).
# Enter short when price breaks below 1d Donchian(20) low with volume > 2x 20-bar average and price < 4h EMA34 (downtrend).
# Uses discrete position sizing (0.20) to limit drawdown. Target: 60-150 trades over 4 years.
# 4h EMA34 filters counter-trend noise, 1d Donchian provides structure, volume confirms momentum.
# Works in bull (breakouts with trend) and bear (failed breaks via exits) markets.

name = "1h_Donchian20_4hEMA34_Volume_Breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA34 trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA34
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA34 to 1h timeframe
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for Donchian channels (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    def donchian_channels(high, low, length=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(length-1, len(high)):
            upper[i] = np.max(high[i-length+1:i+1])
            lower[i] = np.min(low[i-length+1:i+1])
        return upper, lower
    
    donchian_upper_1d, donchian_lower_1d = donchian_channels(high_1d, low_1d, 20)
    
    # Align 1d Donchian to 1h timeframe
    donchian_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    
    # Calculate 1h volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(donchian_upper_1d_aligned[i]) or 
            np.isnan(donchian_lower_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions with volume confirmation and trend filter
        long_breakout = close[i] > donchian_upper_1d_aligned[i] and volume_confirm[i] and close[i] > ema_34_4h_aligned[i]
        short_breakout = close[i] < donchian_lower_1d_aligned[i] and volume_confirm[i] and close[i] < ema_34_4h_aligned[i]
        
        # Exit conditions: opposite Donchian level
        long_exit = close[i] < donchian_lower_1d_aligned[i]
        short_exit = close[i] > donchian_upper_1d_aligned[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals