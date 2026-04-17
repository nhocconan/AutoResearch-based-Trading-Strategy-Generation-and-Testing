#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper AND 12h EMA34 is rising AND volume > 1.5x average.
Short when price breaks below Donchian lower AND 12h EMA34 is falling AND volume > 1.5x average.
Exit when price touches Donchian middle line or opposite breakout occurs.
Uses 12h EMA34 for trend alignment to avoid counter-trend whipsaws in bear markets.
Target: 75-200 total trades over 4 years (19-50/year) on BTC/ETH/SOL.
Donchian provides objective breakout levels, EMA34 filters trend direction, volume confirms conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on primary timeframe (4h)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle_line = (highest_high + lowest_low) / 2.0
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, 34)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(avg_volume[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        middle = middle_line[i]
        ema34 = ema34_12h_aligned[i]
        prev_ema34 = ema34_12h_aligned[i-1] if i > 0 else ema34
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above upper band AND EMA34 rising AND volume confirmed
            if price > upper and ema34 > prev_ema34 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND EMA34 falling AND volume confirmed
            elif price < lower and ema34 < prev_ema34 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches middle line OR price breaks below lower band (contrary signal)
            if price <= middle or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches middle line OR price breaks above upper band (contrary signal)
            if price >= middle or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0