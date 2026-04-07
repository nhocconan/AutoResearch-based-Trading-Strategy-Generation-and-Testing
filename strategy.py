#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Donchian Breakout with Volume Confirmation and ADX Trend Filter
# Hypothesis: Price breaking out of daily Donchian channels (20-period high/low)
# with volume confirmation and ADX > 25 (trending market) captures strong momentum.
# Volume ensures institutional participation. ADX filter avoids ranging markets.
# Works in both bull and bear markets by trading breakouts in direction of trend.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_daily_donchian_breakout_volume_adx_v1"
timeframe = "4h"
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
    
    daily_high_series = pd.Series(daily_high)
    daily_low_series = pd.Series(daily_low)
    donchian_high = daily_high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = daily_low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed daily bars (avoid look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    if len(donchian_high) > 1:
        donchian_high[0] = donchian_high[1]
        donchian_low[0] = donchian_low[1]
    else:
        donchian_high[0] = 0
        donchian_low[0] = 0
    
    # Calculate ADX (14-period) on daily timeframe
    # True Range
    tr1 = np.abs(np.diff(daily_high))
    tr2 = np.abs(np.diff(daily_low))
    tr3 = np.abs(daily_high[1:] - daily_low[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((daily_high[1:] - daily_high[:-1]) > (daily_low[:-1] - daily_low[1:]), 
                       np.maximum(daily_high[1:] - daily_high[:-1], 0), 0)
    dm_minus = np.where((daily_low[:-1] - daily_low[1:]) > (daily_high[1:] - daily_high[:-1]), 
                        np.maximum(daily_low[:-1] - daily_low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = np.where(tr_sum != 0, 100 * dm_plus_sum / tr_sum, 0)
    di_minus = np.where(tr_sum != 0, 100 * dm_minus_sum / tr_sum, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Shift ADX by 1 to avoid look-ahead
    adx = np.roll(adx, 1)
    if len(adx) > 1:
        adx[0] = adx[1]
    else:
        adx[0] = 0
    
    # Align daily data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below daily Donchian low or trend weakens (ADX < 20) or volume filter fails
            if (close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above daily Donchian high or trend weakens (ADX < 20) or volume filter fails
            if (close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above daily Donchian high with volume and strong trend (ADX > 25)
            if (high[i] > donchian_high_aligned[i] and close[i] > donchian_high_aligned[i] and
                adx_aligned[i] > 25 and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below daily Donchian low with volume and strong trend (ADX > 25)
            elif (low[i] < donchian_low_aligned[i] and close[i] < donchian_low_aligned[i] and
                  adx_aligned[i] > 25 and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals