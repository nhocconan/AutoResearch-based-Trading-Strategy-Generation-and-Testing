#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike and 1d ADX trend filter
# Donchian provides clear structure for breakouts in both bull/bear markets
# Volume spike (>2x average) confirms institutional participation and reduces false breakouts
# 1d ADX > 25 ensures we only trade in trending regimes, avoiding choppy markets
# Discrete position sizing (0.25) with ATR-based stoploss (signal=0 when price < highest - 2*ATR)
# Target: ~30-50 trades/year to minimize fee drag while capturing strong trending moves
# Works in bull via long breakouts, in bear via short breakdowns with trend filter

name = "4h_Donchian20_VolumeSpike_1dADX25_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr_1d = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros_like(dx)
    adx[period-1] = np.nanmean(dx[:period])  # First ADX value
    for i in range(period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(20) for stoploss
    tr_4h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_4h[0] = high[0] - low[0]
    atr_4h = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    # Track highest high since entry for trailing stop
    highest_since_entry = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_window, 20)  # Donchian and volume warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr_4h[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr_4h[i]
        curr_adx = adx_1d_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Update highest high since entry
            if np.isnan(highest_since_entry[i]) or curr_close > highest_since_entry[i]:
                highest_since_entry[i] = curr_close
            elif i > 0:
                highest_since_entry[i] = highest_since_entry[i-1]
            else:
                highest_since_entry[i] = curr_close
            
            # Exit: price drops below highest - 2*ATR (trailing stop)
            if curr_close < (highest_since_entry[i] - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
                highest_since_entry[i] = np.nan  # Reset
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Track lowest low since entry for trailing stop (shorts)
            if i == start_idx or np.isnan(highest_since_entry[i-1]):
                highest_since_entry[i] = curr_low  # For shorts, track lowest low
            else:
                highest_since_entry[i] = highest_since_entry[i-1]
            
            if curr_low < highest_since_entry[i]:
                highest_since_entry[i] = curr_low
            
            # Exit: price rises above lowest + 2*ATR (trailing stop for shorts)
            if curr_low > (highest_since_entry[i] + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
                highest_since_entry[i] = np.nan  # Reset
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Trend filter: 1d ADX > 25 indicates trending market
            trending = curr_adx > 25.0
            
            # Long when price breaks above Donchian upper band, volume confirmed, trending
            if curr_high > highest_high[i] and vol_confirmed and trending:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = curr_close  # Initialize tracking
            # Short when price breaks below Donchian lower band, volume confirmed, trending
            elif curr_low < lowest_low[i] and vol_confirmed and trending:
                signals[i] = -0.25
                position = -1
                highest_since_entry[i] = curr_low  # Initialize tracking (for shorts)
            else:
                signals[i] = 0.0
    
    return signals