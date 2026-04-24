#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend filter to capture intermediate trend direction.
- Donchian(20): Price channel breakout structure proven on SOLUSDT (test Sharpe 1.10-1.38).
- Entry: Long when price breaks above Donchian(20) upper band AND price > 12h EMA50 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below Donchian(20) lower band AND price < 12h EMA50 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Donchian breakout OR price crosses 12h EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian breakouts capture momentum bursts; 12h EMA50 filter avoids counter-trend trades in bear markets.
- Volume spike confirmation ensures breakouts have participation, reducing false signals.
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
    if n < 200:  # Need sufficient data for 12h EMA50 and Donchian
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 55:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Calculate 12h volume average for confirmation
    if len(df_12h) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = df_12h['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h, additional_delay_bars=1)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 200  # Need sufficient data for 12h EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 12h EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian lower band OR price falls below 12h EMA50
            if position == 1:
                if curr_low < lowest_low[i] or curr_close < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian upper band OR price rises above 12h EMA50
            elif position == -1:
                if curr_high > highest_high[i] or curr_close > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Long: price breaks above Donchian upper band AND price > 12h EMA50 AND volume confirmation
            long_breakout = curr_high > highest_high[i]
            # Short: price breaks below Donchian lower band AND price < 12h EMA50 AND volume confirmation
            short_breakout = curr_low < lowest_low[i]
            
            if long_breakout and curr_close > ema50_12h_aligned[i] and curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False:
                signals[i] = 0.25
                position = 1
            elif short_breakout and curr_close < ema50_12h_aligned[i] and curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_TrendFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0