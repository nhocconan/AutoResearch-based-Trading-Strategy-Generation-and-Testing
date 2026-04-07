#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout + Daily Trend + Volume Confirmation
# Hypothesis: 12h Donchian(20) breakouts with daily trend filter and volume confirmation
# capture institutional breakouts in both bull and bear markets. Daily EMA50 filters
# trend direction, volume > 1.8x 20-period average confirms institutional participation.
# Designed for low trade frequency (12-37/year) to minimize fee drag.
# Works in bull via long breakouts above 20-period high, in bear via short breakdowns
# below 20-period low, both requiring daily trend alignment and volume confirmation.

name = "12h_donchian20_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    # Highest high and lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily trend filter: EMA50 of daily close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR daily trend turns bearish
            if close[i] < donchian_low[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR daily trend turns bullish
            if close[i] > donchian_high[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Determine trend: price vs daily EMA50
                if close[i] > ema_50_1d_aligned[i]:  # Uptrend
                    # Long: price breaks above Donchian high (breakout)
                    if close[i] > donchian_high[i] and (i == 20 or close[i-1] <= donchian_high[i-1]):
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    # Short: price breaks below Donchian low (breakdown)
                    if close[i] < donchian_low[i] and (i == 20 or close[i-1] >= donchian_low[i-1]):
                        position = -1
                        signals[i] = -0.25
    
    return signals