#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w EMA34 trend filter with Donchian(20) breakout and volume confirmation.
# Enter long when price breaks above Donchian(20) high with volume > 1.5x 20-bar average and price > 1w EMA34 (uptrend).
# Enter short when price breaks below Donchian(20) low with volume > 1.5x 20-bar average and price < 1w EMA34 (downtrend).
# Uses discrete position sizing (0.25) to limit drawdown. Target: 30-100 trades over 4 years.
# Donchian provides objective breakout levels, volume confirms momentum, 1w EMA34 filters counter-trend noise.
# Works in bull (breakouts with trend) and bear (failed breaks via exits) markets.

name = "1d_Donchian20_1wEMA34_Volume_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Donchian channels (20)
    def donchian_channels(high, low, length=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(length-1, len(high)):
            upper[i] = np.max(high[i-length+1:i+1])
            lower[i] = np.min(low[i-length+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Calculate 1d volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions with volume confirmation and trend filter
        long_breakout = close[i] > donchian_upper[i] and volume_confirm[i] and close[i] > ema_34_1w_aligned[i]
        short_breakout = close[i] < donchian_lower[i] and volume_confirm[i] and close[i] < ema_34_1w_aligned[i]
        
        # Exit conditions: opposite Donchian level
        long_exit = close[i] < donchian_lower[i]
        short_exit = close[i] > donchian_upper[i]
        
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