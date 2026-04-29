#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 1d ADX > 25 AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian(20) low AND 1d ADX > 25 AND volume > 1.5x 20-bar avg
# Exit when price crosses opposite Donchian level or ADX drops below 20 (trend weak)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing trends.
# Target: 75-200 trades total over 4 years (19-50/year) on 4h.
# Donchian channels provide objective breakout levels; 1d ADX filters for trending markets only.
# Volume confirmation ensures breakouts have institutional participation.
# Works in bull markets (trend continuation) and bear markets (strong trends down).

name = "4h_Donchian20_1dADX25_VolumeConfirm_v1"
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no prior close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = np.nan
    dm_minus[0] = np.nan
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # Avoid division by zero
    dm_plus_div = np.where(atr_1d != 0, dm_plus_smooth / atr_1d, 0)
    dm_minus_div = np.where(atr_1d != 0, dm_minus_smooth / atr_1d, 0)
    
    dx = np.where((dm_plus_div + dm_minus_div) != 0,
                  np.abs(dm_plus_div - dm_minus_div) / (dm_plus_div + dm_minus_div) * 100, 0)
    adx_1d = wilders_smoothing(dx, period)
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian(20) on 4h data
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30) + 1  # Donchian warmup + ADX warmup + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        adx = adx_1d_aligned[i]
        
        # Donchian levels
        dc_up = dc_upper[i]
        dc_low = dc_lower[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low OR ADX drops below 20 (trend weak)
            if curr_close < dc_low or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high OR ADX drops below 20 (trend weak)
            if curr_close > dc_up or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND ADX > 25 AND volume confirmation
            if curr_close > dc_up and adx > 25 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND ADX > 25 AND volume confirmation
            elif curr_close < dc_low and adx > 25 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals