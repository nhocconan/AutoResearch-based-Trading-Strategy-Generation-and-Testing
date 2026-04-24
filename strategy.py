#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for trend filter (price above/below EMA50).
- Entry: Long when price breaks above Donchian(20) high AND price > 12h EMA50 AND volume > 1.5x 20-period average volume.
         Short when price breaks below Donchian(20) low AND price < 12h EMA50 AND volume > 1.5x 20-period average volume.
- Exit: Opposite Donchian breakout OR price crosses 12h EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear breakout levels with built-in volatility adaptation.
- Volume confirmation ensures breakouts have participation.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Donchian channels (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for Donchian/EMA/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 12h EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian low OR price falls below 12h EMA50
            if position == 1:
                if curr_close < donchian_low[i] or curr_close < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR price rises above 12h EMA50
            elif position == -1:
                if curr_close > donchian_high[i] or curr_close > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and trend alignment
        if position == 0:
            # Long: price breaks above Donchian high AND above 12h EMA50 AND volume confirmation
            if curr_close > donchian_high[i] and curr_close > ema50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 12h EMA50 AND volume confirmation
            elif curr_close < donchian_low[i] and curr_close < ema50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0