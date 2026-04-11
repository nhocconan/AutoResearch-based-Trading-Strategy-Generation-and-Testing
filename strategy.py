#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_adx_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return signals
    
    # Calculate ADX (14) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Plus Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    # Minus Directional Movement
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        smoothed = np.zeros_like(data)
        smoothed[period-1] = np.nansum(data[:period])  # Initial sum
        for i in range(period, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
        return smoothed
    
    tr14 = wilder_smooth(tr, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros_like(dx)
    adx[27] = np.nanmean(dx[14:28])  # First ADX value (average of first 14 DX)
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h Donchian channel (20-period)
    def donchian_channel(high, low, period):
        upper = np.zeros_like(high)
        lower = np.zeros_like(low)
        for i in range(len(high)):
            if i < period - 1:
                upper[i] = np.nan
                lower[i] = np.nan
            else:
                upper[i] = np.max(high[i-period+1:i+1])
                lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper, lower = donchian_channel(high, low, 20)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        
        # ADX threshold for trending market (ADX > 25)
        trending = adx_val > 25
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Breakout conditions
        long_breakout = price_high > upper[i]
        short_breakout = price_low < lower[i]
        
        # Trading logic
        if trending and volume_confirmed:
            if long_breakout and position != 1:
                position = 1
                signals[i] = 0.25
            elif short_breakout and position != -1:
                position = -1
                signals[i] = -0.25
        else:
            # Exit positions in non-trending markets
            if position == 1:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                position = 0
                signals[i] = 0.0
        
        # Maintain current position if no action taken
        if position == 1 and signals[i] == 0.0:
            signals[i] = 0.25
        elif position == -1 and signals[i] == 0.0:
            signals[i] = -0.25
    
    return signals

# Hypothesis: ADX-filtered Donchian breakout strategy on 4h timeframe.
# Uses 14-period ADX on daily timeframe to filter for trending markets (ADX > 25).
# Enters long when price breaks above 20-period Donchian upper band with volume confirmation (>1.5x average volume) in trending markets.
# Enters short when price breaks below 20-period Donchian lower band with volume confirmation in trending markets.
# Exits positions when market becomes non-trending (ADX <= 25) to avoid whipsaws in ranging markets.
# Works in both bull and bear markets by capturing trends while avoiding false breakouts in sideways markets.
# Volume confirmation reduces false breakouts. ADX filter ensures we only trade during strong trends.
# Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag.