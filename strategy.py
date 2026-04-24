#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for trend filter (price above/below EMA50).
- Entry: Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND volume > 1.5x 20-bar average volume.
         Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND volume > 1.5x 20-bar average volume.
- Exit: Opposite Donchian breakout OR price crosses 1d EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear breakout levels with built-in trend following.
- EMA50 on 1d filter ensures trades align with higher timeframe trend.
- Volume confirmation filters out false breakouts.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def donchian_channels(high, low, period):
    """Calculate Donchian channels: upper=max(high,period), lower=min(low,period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Donchian(20) channels
    donchian_period = 20
    upper_channel, lower_channel = donchian_channels(high, low, donchian_period)
    
    # Volume confirmation: volume > 1.5x 20-bar average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, donchian_period)  # Need sufficient data for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1d EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below lower Donchian OR price falls below 1d EMA50
            if position == 1:
                if curr_close < lower_channel[i] or curr_close < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above upper Donchian OR price rises above 1d EMA50
            elif position == -1:
                if curr_close > upper_channel[i] or curr_close > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout + trend filter + volume confirmation
        if position == 0:
            # Long: price breaks above upper Donchian AND bullish 1d trend AND volume spike
            if curr_close > upper_channel[i] and curr_close > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND bearish 1d trend AND volume spike
            elif curr_close < lower_channel[i] and curr_close < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0