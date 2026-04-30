#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Donchian(20) breakout + 12h ADX25 trend filter + volume confirmation
# Donchian channels provide clear breakout levels; 12h ADX>25 filters for trending markets only.
# Volume spike (1.8x 20-period average) confirms breakout validity.
# Uses 6h timeframe for signal generation, 12h for trend filter. Discrete sizing 0.25 to minimize fee churn.
# Target: 80-120 total trades over 4 years (20-30/year).

name = "6h_Donchian20_Breakout_12hADX25_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 6h data ONCE before loop for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (20-period) using prior 6h bar
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian upper/lower bands (20-period)
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (wait for completed 6h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    
    # Load 12h data ONCE before loop for ADX25 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 12h ADX (14-period) with DI+ and DI-
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range (TR)
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate +DM and -DM
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
            else:
                result[i] = np.nan
        return result
    
    atr = wilder_smooth(tr, 14)
    atr[np.isnan(atr)] = 0  # Replace NaN with 0 for safety
    
    plus_di = 100 * wilder_smooth(plus_dm, 14) / (atr + 1e-10)
    minus_di = 100 * wilder_smooth(minus_dm, 14) / (atr + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilder_smooth(dx, 14)
    adx[np.isnan(adx)] = 0  # Replace NaN with 0 for safety
    
    # Align ADX to 6h timeframe (wait for completed 12h bar)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_adx = adx_aligned[i]
        curr_upper = donchian_upper_aligned[i]
        curr_lower = donchian_lower_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and ADX > 25 (trending market)
            if volume_spike and curr_adx > 25:
                # Bullish entry: price breaks above Donchian upper band
                if curr_close > curr_upper:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian lower band
                elif curr_close < curr_lower:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below Donchian lower band or ADX < 20 (trend weakens)
            if curr_close < curr_lower or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian upper band or ADX < 20 (trend weakens)
            if curr_close > curr_upper or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals