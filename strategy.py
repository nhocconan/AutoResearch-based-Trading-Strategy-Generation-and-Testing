#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA200 trend filter and 1d volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA200 trend filter and volume spike filter.
- Entry: Long when price breaks above Donchian(20) high AND price > 1d EMA200 AND volume > 1.5 * 20-period average volume.
         Short when price breaks below Donchian(20) low AND price < 1d EMA200 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian breakout OR price crosses 1d EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear structure with fixed lookback period.
- 1d EMA200 provides strong trend filter to avoid counter-trend trades in ranging markets.
- Volume spike confirmation ensures breakouts have conviction and reduces false signals.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def donchian_channels(high, low, period):
    """Calculate Donchian channels (upper, lower)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 1d volume average for spike confirmation
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike_threshold = 1.5 * vol_ma_20
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20, additional_delay_bars=1)
    vol_spike_threshold_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_threshold, additional_delay_bars=1)
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    upper_20, lower_20 = donchian_channels(df_12h['high'].values, df_12h['low'].values, 20)
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20, additional_delay_bars=1)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for 1d EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(vol_spike_threshold_aligned[i]) or np.isnan(upper_20_aligned[i]) or
            np.isnan(lower_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Exit long: price breaks below 12h Donchian lower OR price falls below 1d EMA200
            if position == 1:
                if curr_close < lower_20_aligned[i] or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above 12h Donchian upper OR price rises above 1d EMA200
            elif position == -1:
                if curr_close > upper_20_aligned[i] or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and trend filter
        if position == 0:
            # Long: price breaks above 12h Donchian upper AND volume spike AND bullish 1d trend
            if curr_close > upper_20_aligned[i] and curr_volume > vol_spike_threshold_aligned[i] and curr_close > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower AND volume spike AND bearish 1d trend
            elif curr_close < lower_20_aligned[i] and curr_volume > vol_spike_threshold_aligned[i] and curr_close < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA200_TrendFilter_1dVolumeSpike_v1"
timeframe = "12h"
leverage = 1.0