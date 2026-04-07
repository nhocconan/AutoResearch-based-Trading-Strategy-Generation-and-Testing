#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6H Weekly Donchian Breakout with Volume and ADX Filter
# Hypothesis: Weekly Donchian channels provide robust trend-following structure for 6H entries.
# Breakouts above/below weekly high/low with volume confirmation and ADX>25 capture strong trends.
# Works in bull markets (buy breakouts) and bear markets (sell breakdowns).
# Weekly timeframe filters noise, volume confirms institutional interest, ADX avoids ranging markets.
# Target: 12-37 trades/year (50-150 total over 4 years).

name = "6h_weekly_donchian_breakout_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and ADX
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Weekly high/low/close
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high_series = pd.Series(weekly_high)
    weekly_low_series = pd.Series(weekly_low)
    weekly_donchian_high = weekly_high_series.rolling(window=20, min_periods=20).max().values
    weekly_donchian_low = weekly_low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly ADX (14-period)
    # True Range
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = weekly_high - np.roll(weekly_high, 1)
    down_move = np.roll(weekly_low, 1) - weekly_low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth_series(series, period):
        result = np.full_like(series, np.nan)
        if len(series) >= period:
            result[period-1] = np.nansum(series[:period])
            for i in range(period, len(series)):
                result[i] = result[i-1] - (result[i-1] / period) + series[i]
        return result
    
    atr = smooth_series(tr, 14)
    plus_di = 100 * smooth_series(plus_dm, 14) / atr
    minus_di = 100 * smooth_series(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_series(dx, 14)
    
    # Handle edge cases
    atr[:13] = np.nan
    plus_di[:13] = np.nan
    minus_di[:13] = np.nan
    dx[:13] = np.nan
    adx[:27] = np.nan  # ADX needs 28 periods (14+14)
    
    # Align weekly indicators to 6H timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below weekly Donchian low or ADX weakens
            if close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above weekly Donchian high or ADX weakens
            if close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above weekly Donchian high with volume and strong trend
            if (high[i] > donchian_high_aligned[i] and close[i] > donchian_high_aligned[i] and
                vol_filter[i] and adx_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below weekly Donchian low with volume and strong trend
            elif (low[i] < donchian_low_aligned[i] and close[i] < donchian_low_aligned[i] and
                  vol_filter[i] and adx_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals