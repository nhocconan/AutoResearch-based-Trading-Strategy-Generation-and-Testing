#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Breakout with 4h/1d Trend Filter and Volume Confirmation.
# Uses 4h EMA20 for trend direction and 1d Donchian20 breakout for entry timing.
# Volume filter (current volume > 1.5x 20-period average) ensures quality signals.
# Works in bull/bear markets via trend alignment and breakout structure.
# Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_breakout_4h_ema_1d_donchian_v1"
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
    
    # 4h EMA20 for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    for i in range(19, len(close_4h)):
        if i == 19:
            ema_4h[i] = np.mean(close_4h[0:20])
        else:
            ema_4h[i] = (close_4h[i] * 0.0952) + (ema_4h[i-1] * 0.9048)  # alpha=2/(20+1)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d Donchian20 for breakout levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high = np.full(len(high_1d), np.nan)
    donch_low = np.full(len(low_1d), np.nan)
    for i in range(19, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-19:i+1])
        donch_low[i] = np.min(low_1d[i-19:i+1])
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Combined filters
        volume_ok = volume[i] > vol_ma[i] * 1.5
        session_ok = session_filter[i]
        
        if position == 0:  # look for entries
            if volume_ok and session_ok:
                # Long: uptrend (price > 4h EMA20) and breakout above 1d Donchian high
                if close[i] > ema_4h_aligned[i] and close[i] > donch_high_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: downtrend (price < 4h EMA20) and breakdown below 1d Donchian low
                elif close[i] < ema_4h_aligned[i] and close[i] < donch_low_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
        elif position == 1:  # long position
            # Exit: trend reversal or breakdown below 1d Donchian low
            if close[i] < ema_4h_aligned[i] or close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: trend reversal or breakout above 1d Donchian high
            if close[i] > ema_4h_aligned[i] or close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals