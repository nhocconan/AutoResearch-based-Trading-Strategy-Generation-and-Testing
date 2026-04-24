#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for trend filter (price above/below EMA50).
- Entry: Long when price breaks above Donchian(20) high AND 1w EMA50 uptrend AND volume > 1.5x 20-period average.
         Short when price breaks below Donchian(20) low AND 1w EMA50 downtrend AND volume > 1.5x 20-period average.
- Exit: Opposite Donchian breakout OR price crosses 1w EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian channels provide clear breakout levels with built-in stoploss (opposite channel).
- 1w EMA50 filters for primary trend alignment to avoid counter-trend trades.
- Volume spike confirms institutional participation behind breakouts.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~80 total over 4 years (~20/year) based on filtered breakout frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate Donchian(20) channels on 12h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 60)  # Donchian(20), volume MA(20), 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1w EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian low OR price falls below 1w EMA50
            if position == 1:
                if curr_close < donchian_low[i] or curr_close < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR price rises above 1w EMA50
            elif position == -1:
                if curr_close > donchian_high[i] or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume confirmation
        if position == 0:
            # Long: price breaks above Donchian high AND 1w EMA50 uptrend AND volume spike
            if (curr_close > donchian_high[i] and 
                close[i-1] <= donchian_high[i-1] and  # Ensure breakout just occurred
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and  # 1w EMA50 rising
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND 1w EMA50 downtrend AND volume spike
            elif (curr_close < donchian_low[i] and 
                  close[i-1] >= donchian_low[i-1] and  # Ensure breakdown just occurred
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and  # 1w EMA50 falling
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0