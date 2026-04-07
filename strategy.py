#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-day volume confirmation and ATR stoploss
# Long when price breaks above 20-period high with volume > 1.5x 20-period average
# Short when price breaks below 20-period low with volume > 1.5x 20-period average
# Exit when price crosses opposite Donchian level or trailing stoploss at 3*ATR
# Position size: 0.25 (25% of capital)
# Uses 1-day volume for confirmation to filter false breakouts
# Target: 100-200 total trades over 4 years (25-50/year)

name = "12h_donchian20_1d_vol_confirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    avg_volume_1d = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    
    # 12-period Donchian channels (20-period lookback)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # 12-period ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1-day volume average to 12h
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # Trailing stoploss: 3 * ATR from high
            if high[i] < highest_since_entry - 3 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: price crosses below Donchian low (mean reversion)
            elif close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # Trailing stoploss: 3 * ATR from low
            if low[i] > lowest_since_entry + 3 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: price crosses above Donchian high (mean reversion)
            elif close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume confirmation
            volume_confirm = volume[i] > 1.5 * avg_volume_1d_aligned[i]
            
            # Long: price breaks above Donchian high with volume confirmation
            if close[i] > donchian_high[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                highest_since_entry = high[i]
            # Short: price breaks below Donchian low with volume confirmation
            elif close[i] < donchian_low[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                lowest_since_entry = low[i]
    
    return signals