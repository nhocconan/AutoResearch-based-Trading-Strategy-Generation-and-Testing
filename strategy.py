#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20-period) + 12h EMA trend filter + volume confirmation
# Long when price breaks above Donchian upper band AND price > 12h EMA50 AND volume > 1.5x 20-period average volume
# Short when price breaks below Donchian lower band AND price < 12h EMA50 AND volume > 1.5x 20-period average volume
# Exit when price crosses back through Donchian middle (midpoint) OR volume drops below average
# Uses Donchian channels for clear breakout signals, EMA for trend filter, volume for confirmation
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_high + lowest_low) / 2
    
    # Calculate 12h EMA50
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        upper_break = close[i] > highest_high[i]
        lower_break = close[i] < lowest_low[i]
        
        # Volume confirmation
        vol_confirmed = volume[i] > (vol_avg[i] * 1.5)
        
        # Trend filter
        above_ema = close[i] > ema_12h_aligned[i]
        below_ema = close[i] < ema_12h_aligned[i]
        
        # Entry conditions
        long_entry = upper_break and above_ema and vol_confirmed
        short_entry = lower_break and below_ema and vol_confirmed
        
        # Exit conditions: price crosses middle OR volume drops
        exit_long = position == 1 and (close[i] < donchian_middle[i] or volume[i] < vol_avg[i])
        exit_short = position == -1 and (close[i] > donchian_middle[i] or volume[i] < vol_avg[i])
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_ema_volume"
timeframe = "4h"
leverage = 1.0