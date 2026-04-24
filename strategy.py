#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter and volume average.
- Bollinger Bands: identifies low volatility squeezes (BB width < 20th percentile) that precede breakouts.
- Entry: Long when price breaks above upper BB AND BB width is in lowest 20% (squeeze) AND price > 1d EMA50 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below lower BB AND BB width is in lowest 20% (squeeze) AND price < 1d EMA50 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite BB breakout signal or BB width expands above 50% (squeeze end).
- Signal size: 0.25 discrete to minimize fee drag.
- Bollinger squeeze works in both bull and bear markets as it captures volatility contraction/expansion cycles.
- Volume confirmation ensures breakout legitimacy.
- 1d EMA50 provides robust trend filter to avoid counter-trend trades in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands: returns upper, lower, bandwidth."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    bandwidth = (upper - lower) / (sma + 1e-10)  # Avoid division by zero
    return upper, lower, bandwidth

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Bollinger Bands from 6h data (20-period, 2 std)
    bb_upper, bb_lower, bb_width = bollinger_bands(close, 20, 2.0)
    
    # Calculate 20th percentile of BB width for squeeze condition (using expanding window)
    bb_width_percentile = np.zeros_like(bb_width)
    for i in range(len(bb_width)):
        if i < 20:
            bb_width_percentile[i] = np.nan
        else:
            bb_width_percentile[i] = np.percentile(bb_width[max(0, i-50):i+1], 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Need 20 for BB, 50 for 1d EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_width[i]) or
            np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions
        if position != 0:
            # Exit long: price breaks below lower BB OR BB width expands above 50% (squeeze end)
            if position == 1:
                if curr_close < bb_lower[i] or bb_width[i] > bb_width_percentile[i] * 2.5:  # Exit when width > 2.5x 20th percentile
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above upper BB OR BB width expands above 50% (squeeze end)
            elif position == -1:
                if curr_close > bb_upper[i] or bb_width[i] > bb_width_percentile[i] * 2.5:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: BB breakout with squeeze, trend filter, and volume confirmation
        if position == 0:
            # BB breakout signals
            breakout_up = curr_close > bb_upper[i] and prev_close <= bb_upper[i-1]
            breakout_down = curr_close < bb_lower[i] and prev_close >= bb_lower[i-1]
            
            # Squeeze condition: BB width in lowest 20% (volatility contraction)
            squeeze_condition = bb_width[i] <= bb_width_percentile[i]
            
            # Trend filter: price vs 1d EMA50
            long_trend = curr_close > ema50_1d_aligned[i]
            short_trend = curr_close < ema50_1d_aligned[i]
            
            # Volume confirmation: current volume > 2.0 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 2.0 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            if breakout_up and squeeze_condition and long_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif breakout_down and squeeze_condition and short_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_Breakout_1dEMA50_TrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0