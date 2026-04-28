#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakout with volume confirmation and ADX trend filter.
# Enter long when price breaks above 1d Donchian upper (20) with volume spike and ADX > 25 (trending).
# Enter short when price breaks below 1d Donchian lower (20) with volume spike and ADX > 25.
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 20-50 trades/year.
# Donchian provides structure from higher timeframe, volume confirms breakout strength, ADX filter avoids ranging markets.
# Works in bull (breakouts with trend) and bear (failed breaks reverse via exits) markets.

name = "4h_Donchian20_1d_Volume_ADXFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period, using previous bar to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    donchian_high = np.full(n_1d, np.nan)
    donchian_low = np.full(n_1d, np.nan)
    
    for i in range(20, n_1d):  # Start at 20 to ensure 20-bar lookback
        # Use previous 20 bars (i-20 to i-1) to avoid look-ahead
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Forward fill Donchian levels
    donchian_high = pd.Series(donchian_high).ffill().values
    donchian_low = pd.Series(donchian_low).ffill().values
    
    # Align 1d indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 4h ADX (14) for trend filter
    def calculate_adx(high, low, close, length=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0  # First value has no previous close
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/length)
        atr = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        # Initial values (simple average)
        atr[length] = np.mean(tr[1:length+1])
        dm_plus_smooth[length] = np.mean(dm_plus[1:length+1])
        dm_minus_smooth[length] = np.mean(dm_minus[1:length+1])
        
        # Wilder's smoothing: today = (yesterday * (length-1) + today) / length
        for i in range(length+1, len(tr)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (length-1) + dm_plus[i]) / length
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (length-1) + dm_minus[i]) / length
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = np.zeros_like(dx)
        
        # Initial ADX value (simple average of first 'length' DX values)
        adx[2*length] = np.mean(dx[length:2*length+1])
        
        # Wilder's smoothing for ADX
        for i in range(2*length+1, len(dx)):
            adx[i] = (adx[i-1] * (length-1) + dx[i]) / length
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    adx_trending = adx > 25  # Trending regime when ADX > 25
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions with volume confirmation and ADX filter
        long_breakout = close[i] > donchian_high_aligned[i] and volume_spike[i] and adx_trending[i]
        short_breakout = close[i] < donchian_low_aligned[i] and volume_spike[i] and adx_trending[i]
        
        # Exit conditions: opposite Donchian level
        long_exit = close[i] < donchian_low_aligned[i]
        short_exit = close[i] > donchian_high_aligned[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals