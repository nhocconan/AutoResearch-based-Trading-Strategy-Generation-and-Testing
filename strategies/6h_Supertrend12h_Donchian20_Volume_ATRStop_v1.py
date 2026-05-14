#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend for trend direction and 6h price action for entry timing.
# Long when: 6h close > 12h Supertrend (uptrend) AND 6h close > 6h Donchian(20) upper band with volume > 1.5x 20-bar average
# Short when: 6h close < 12h Supertrend (downtrend) AND 6h close < 6h Donchian(20) lower band with volume > 1.5x 20-bar average
# Exit via ATR(14) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses 12h Supertrend for robust trend filtering (avoids whipsaw), 6h Donchian breakouts for entry precision, volume for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_Supertrend12h_Donchian20_Volume_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Supertrend for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ATR for Supertrend (10 period)
    tr1_12h = high_12h[1:] - low_12h[1:]
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation (10, 3.0)
    hl2_12h = (high_12h + low_12h) / 2
    upperband_12h = hl2_12h + 3.0 * atr_12h
    lowerband_12h = hl2_12h - 3.0 * atr_12h
    
    supertrend_12h = np.full_like(close_12h, np.nan)
    direction_12h = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(supertrend_12h[i-1]):
            # Initialize
            supertrend_12h[i] = lowerband_12h[i]
            direction_12h[i] = 1
        else:
            if close_12h[i] > upperband_12h[i-1]:
                supertrend_12h[i] = lowerband_12h[i]
                direction_12h[i] = 1
            elif close_12h[i] < lowerband_12h[i-1]:
                supertrend_12h[i] = upperband_12h[i]
                direction_12h[i] = -1
            else:
                supertrend_12h[i] = supertrend_12h[i-1]
                direction_12h[i] = direction_12h[i-1]
                
                # Adjust bands
                if direction_12h[i] == 1 and lowerband_12h[i] < lowerband_12h[i-1]:
                    lowerband_12h[i] = lowerband_12h[i-1]
                if direction_12h[i] == -1 and upperband_12h[i] > upperband_12h[i-1]:
                    upperband_12h[i] = upperband_12h[i-1]
    
    # Align 12h Supertrend and direction to 6h timeframe (completed 12h bar only)
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # 6h Donchian(20) for entry timing
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().shift(1).values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().shift(1).values
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for Donchian and ATR calculations)
    start_idx = max(donchian_window, 20) + 5
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(supertrend_12h_aligned[i]) or np.isnan(direction_12h_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: 12h Supertrend uptrend AND price breaks above 6h Donchian upper with volume spike
            if direction_12h_aligned[i] == 1 and close[i] > upper_channel[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: 12h Supertrend downtrend AND price breaks below 6h Donchian lower with volume spike
            elif direction_12h_aligned[i] == -1 and close[i] < lower_channel[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.5 * ATR
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.5 * ATR
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals