#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Donchian Breakout with Volume and ADX Filter
# Hypothesis: Price breaking out of daily Donchian channels (20-period high/low)
# with volume confirmation and daily ADX trend filter captures strong momentum moves.
# Uses 12h timeframe to reduce noise and transaction costs, targeting 12-37 trades/year.
# Works in bull markets (buy breakouts above daily high) and bear markets
# (sell breakouts below daily low) by following the trend on higher timeframe.

name = "12h_daily_donchian_breakout_volume_adx_v1"
timeframe = "12h"
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
    
    # Get daily data for Donchian channels and ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period high/low)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate rolling max/min for Donchian channels
    daily_high_series = pd.Series(daily_high)
    daily_low_series = pd.Series(daily_low)
    donchian_high = daily_high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = daily_low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed daily bars (avoid look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    
    # Handle first element
    if len(donchian_high) > 1:
        donchian_high[0] = donchian_high[1]
        donchian_low[0] = donchian_low[1]
    else:
        donchian_high[0] = 0
        donchian_low[0] = 0
    
    # Calculate daily ADX (14-period)
    # True Range
    tr1 = np.abs(daily_high[1:] - daily_low[1:])
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((daily_high[1:] - daily_high[:-1]) > (daily_low[:-1] - daily_low[1:]),
                       np.maximum(daily_high[1:] - daily_high[:-1], 0), 0)
    dm_minus = np.where((daily_low[:-1] - daily_low[1:]) > (daily_high[1:] - daily_high[:-1]),
                        np.maximum(daily_low[:-1] - daily_low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR and DM
    tr_period = 14
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    # Wilder's smoothing (first value is simple average)
    if len(tr) >= tr_period:
        atr[tr_period-1] = np.nanmean(tr[1:tr_period])
        dm_plus_smooth[tr_period-1] = np.nanmean(dm_plus[1:tr_period])
        dm_minus_smooth[tr_period-1] = np.nanmean(dm_minus[1:tr_period])
        
        for i in range(tr_period, len(tr)):
            atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period - 1) + dm_plus[i]) / tr_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period - 1) + dm_minus[i]) / tr_period
    
    # Avoid division by zero
    dm_plus_smooth = np.where(atr == 0, 0, dm_plus_smooth)
    dm_minus_smooth = np.where(atr == 0, 0, dm_minus_smooth)
    
    # DI+ and DI-
    di_plus = np.where(atr == 0, 0, 100 * dm_plus_smooth / atr)
    di_minus = np.where(atr == 0, 0, 100 * dm_minus_smooth / atr)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) == 0, 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus))
    adx = np.full_like(dx, np.nan)
    
    # Wilder's smoothing for ADX
    if len(dx) >= tr_period:
        adx[tr_period-1] = np.nanmean(dx[1:tr_period])
        for i in range(tr_period, len(dx)):
            adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    # Align daily data to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume filter: volume > 1.5x 50-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: price falls below daily Donchian low or trend weakens
            if close[i] < donchian_low_aligned[i] or not strong_trend or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above daily Donchian high or trend weakens
            if close[i] > donchian_high_aligned[i] or not strong_trend or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above daily Donchian high with volume and strong trend
            if (high[i] > donchian_high_aligned[i] and close[i] > donchian_high_aligned[i] and
                strong_trend and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below daily Donchian low with volume and strong trend
            elif (low[i] < donchian_low_aligned[i] and close[i] < donchian_low_aligned[i] and
                  strong_trend and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals