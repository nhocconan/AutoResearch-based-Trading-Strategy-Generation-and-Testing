#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe strategy using 4h Donchian breakout for direction and 1h volume spike for entry timing.
# 4h Donchian channels provide trend direction (breakout above/below 20-period high/low).
# 1h volume spike (>2x 20-period average) confirms institutional participation.
# Trades only during active London/NY session (08-20 UTC) to avoid low-volume noise.
# Fixed position size of 0.20 to control risk. Target: 60-150 trades over 4 years.

name = "1h_donchian20_vol_spike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h Donchian channels (20-period high/low) - calculated once before loop
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels
    donchian_high = np.full(len(high_4h), np.nan)
    donchian_low = np.full(len(low_4h), np.nan)
    
    for i in range(19, len(high_4h)):  # 20-period lookback
        donchian_high[i] = np.max(high_4h[i-19:i+1])
        donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # Align to 1h timeframe (shifted by 1 bar for no look-ahead)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1h volume spike filter (>2x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):  # 20-period lookback
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # Need 20 periods for Donchian and volume MA
    
    for i in range(start, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Skip if Donchian data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low or stoploss
            if (close[i] < donchian_low_aligned[i] or 
                close[i] < entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high or stoploss
            if (close[i] > donchian_high_aligned[i] or 
                close[i] > entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Donchian breakout with volume spike
            if volume_spike[i]:
                # Long: breakout above Donchian high
                if close[i] > donchian_high_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: breakout below Donchian low
                elif close[i] < donchian_low_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals