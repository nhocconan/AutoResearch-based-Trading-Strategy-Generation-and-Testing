#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with 1d EMA200 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA200 trend filter to capture major trend direction.
- Donchian channel (20-period): Identifies breakouts above recent highs or below recent lows.
- Entry: Long when price breaks above Donchian upper band AND price > 1d EMA200 AND volume > 1.5 * 20-period average volume.
         Short when price breaks below Donchian lower band AND price < 1d EMA200 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian breakout OR price crosses 1d EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian breakouts capture momentum, effective in both trending and ranging markets with volume confirmation.
- 1d EMA200 provides strong long-term trend filter to avoid counter-trend trades.
- Volume confirmation ensures breakouts have participation, reducing false signals.
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need sufficient data for 1d EMA200
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 205:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d, additional_delay_bars=1)
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 200  # Need sufficient data for 1d EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian lower band OR price falls below 1d EMA200
            if position == 1:
                if curr_low < lowest_low[i] or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian upper band OR price rises above 1d EMA200
            elif position == -1:
                if curr_high > highest_high[i] or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Long: price breaks above Donchian upper band AND price > 1d EMA200 AND volume confirmation
            long_breakout = curr_high > highest_high[i]
            long_trend = curr_close > ema200_1d_aligned[i]
            long_volume = curr_volume > 1.5 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False
            
            # Short: price breaks below Donchian lower band AND price < 1d EMA200 AND volume confirmation
            short_breakout = curr_low < lowest_low[i]
            short_trend = curr_close < ema200_1d_aligned[i]
            short_volume = curr_volume > 1.5 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False
            
            if long_breakout and long_trend and long_volume:
                signals[i] = 0.25
                position = 1
            elif short_breakout and short_trend and short_volume:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA200_TrendFilter_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0