#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter and ATR-based volume spike detection.
- Entry: Long when price breaks above Donchian(20) high AND close > 1d EMA50 AND volume > 1.5 * 20-period average volume.
         Short when price breaks below Donchian(20) low AND close < 1d EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian breakout OR price crosses 1d EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian channels provide clear breakout levels with built-in volatility adaptation.
- 1d EMA50 filters for higher-timeframe trend alignment to avoid counter-trend trades.
- Volume confirmation ensures breakouts have institutional participation.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~80 total over 4 years (~20/year) based on Donchian breakout frequency with filters.
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
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike threshold (using 20-period ATR)
    tr1 = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1)) - np.minimum(df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr1[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # First TR
    atr20_1d = pd.Series(tr1).rolling(window=20, min_periods=20).mean().values
    atr20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr20_1d, additional_delay_bars=1)
    
    # Calculate 20-period average volume for volume confirmation (1d timeframe)
    avg_vol_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20_1d, additional_delay_bars=1)
    
    # Calculate 12h Donchian channels (20-period)
    upper_12h, lower_12h = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for Donchian and 1d indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr20_1d_aligned[i]) or
            np.isnan(avg_vol_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume (1d aligned)
        volume_confirmed = curr_volume > 1.5 * avg_vol_20_1d_aligned[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1d EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian low OR price falls below 1d EMA50
            if position == 1:
                if curr_low < lower_12h[i] or curr_close < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR price rises above 1d EMA50
            elif position == -1:
                if curr_high > upper_12h[i] or curr_close > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and trend alignment
        if position == 0:
            # Long: price breaks above Donchian high AND volume confirmed AND bullish 1d trend
            if curr_high > upper_12h[i] and volume_confirmed and curr_close > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume confirmed AND bearish 1d trend
            elif curr_low < lower_12h[i] and volume_confirmed and curr_close < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeConfirm_1dEMA50_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0