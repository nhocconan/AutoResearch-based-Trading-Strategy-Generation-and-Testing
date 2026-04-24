#!/usr/bin/env python3
"""
Hypothesis: 4h Volume Spike + Donchian(20) Breakout + 12h EMA34 Trend Filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for trend filter (price above/below EMA34).
- Entry: Long when price breaks above Donchian(20) high AND volume > 1.5x 20-bar avg volume AND price > 12h EMA34.
         Short when price breaks below Donchian(20) low AND volume > 1.5x 20-bar avg volume AND price < 12h EMA34.
- Exit: Opposite Donchian breakout OR price crosses 12h EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear breakout levels with built-in trend following.
- Volume confirmation ensures breakouts have conviction.
- 12h EMA34 filter avoids counter-trend trades, improving win rate in choppy markets.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~150 total over 4 years (~37/year) based on Donchian breakout frequency with volume and trend filters.
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
    
    # Calculate 12h trend filter: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h, additional_delay_bars=1)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 12h EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian low OR price falls below 12h EMA34
            if position == 1:
                if curr_close < donchian_low[i] or curr_close < ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR price rises above 12h EMA34
            elif position == -1:
                if curr_close > donchian_high[i] or curr_close > ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: All aligned in same direction
        if position == 0:
            # Long: Donchian breakout up AND volume spike AND bullish 12h trend
            if curr_close > donchian_high[i] and volume_spike[i] and curr_close > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND volume spike AND bearish 12h trend
            elif curr_close < donchian_low[i] and volume_spike[i] and curr_close < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_VolumeSpike_Donchian20_12hEMA34_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0