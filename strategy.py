#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend for trend direction and 6h Donchian(20) breakout with volume confirmation.
# Enter long when price breaks above Donchian(20) high, volume > 1.5x 20-bar average, and 12h Supertrend is bullish.
# Enter short when price breaks below Donchian(20) low, volume > 1.5x 20-bar average, and 12h Supertrend is bearish.
# Uses discrete position sizing (0.25) to limit drawdown. Target: 50-150 trades over 4 years.
# Supertrend provides robust trend filtering, Donchian gives objective breakout levels, volume confirms momentum.
# Works in bull (breakouts with trend) and bear (failed breaks via exits) markets.

name = "6h_Supertrend12h_Donchian20_Breakout_Volume_v1"
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
    
    # Get 12h data for Supertrend trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(10)
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_12h + low_12h) / 2
    upper_basic = hl2 + 3.0 * atr_10
    lower_basic = hl2 - 3.0 * atr_10
    
    # Final Upper and Lower Bands
    upper_final = np.full_like(upper_basic, np.nan)
    lower_final = np.full_like(lower_basic, np.nan)
    for i in range(1, len(upper_basic)):
        if (close_12h[i-1] <= upper_final[i-1]) or (np.isnan(upper_final[i-1])):
            upper_final[i] = min(upper_basic[i], upper_final[i-1])
        else:
            upper_final[i] = upper_basic[i]
            
        if (close_12h[i-1] >= lower_final[i-1]) or (np.isnan(lower_final[i-1])):
            lower_final[i] = max(lower_basic[i], lower_final[i-1])
        else:
            lower_final[i] = lower_basic[i]
    
    # Supertrend direction
    supertrend = np.full_like(close_12h, np.nan)
    for i in range(1, len(close_12h)):
        if np.isnan(supertrend[i-1]):
            supertrend[i] = upper_final[i]
        elif close_12h[i] <= supertrend[i-1]:
            supertrend[i] = upper_final[i]
        else:
            supertrend[i] = lower_final[i]
    
    # Supertrend trend direction (1 = bullish, -1 = bearish)
    trend_dir = np.where(close_12h > supertrend, 1, -1)
    
    # Align 12h Supertrend trend to 6h timeframe
    trend_dir_aligned = align_htf_to_ltf(prices, df_12h, trend_dir)
    
    # Calculate 6h Donchian channels (20)
    def donchian_channels(high, low, length=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(length-1, len(high)):
            upper[i] = np.max(high[i-length+1:i+1])
            lower[i] = np.min(low[i-length+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Calculate 6h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_dir_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions with volume confirmation and trend filter
        long_breakout = close[i] > donchian_upper[i] and volume_confirm[i] and trend_dir_aligned[i] == 1
        short_breakout = close[i] < donchian_lower[i] and volume_confirm[i] and trend_dir_aligned[i] == -1
        
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