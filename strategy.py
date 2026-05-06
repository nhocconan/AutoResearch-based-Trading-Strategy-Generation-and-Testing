#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with volume confirmation and 1w ADX trend filter
# - Long when price breaks above 1d Donchian high (20) with volume > 1.5x 20-bar average
# - Short when price breaks below 1d Donchian low (20) with volume > 1.5x 20-bar average
# - Only take trades when 1w ADX > 25 (strong trend) to avoid whipsaws in ranging markets
# - Exit when price returns to 1d Donchian midpoint (mean reversion within the channel)
# - Position size: 0.25 (25% of capital) to manage drawdown
# - Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency

name = "4h_1dDonchian20_1wADX_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period high/low)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 1d Donchian levels to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_4h = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First value has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (first 14 periods)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[1:15])
        dm_plus_smooth[13] = np.mean(dm_plus[1:15])
        dm_minus_smooth[13] = np.mean(dm_minus[1:15])
        
        # Wilder's smoothing
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # Avoid division by zero
    dmi_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    dmi_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    dx = np.where((dmi_plus + dmi_minus) != 0, np.abs(dmi_plus - dmi_minus) / (dmi_plus + dmi_minus) * 100, 0)
    
    # ADX is smoothed DX
    adx = np.zeros_like(dx)
    if len(dx) >= 14:
        adx[27] = np.mean(dx[14:28])  # First ADX after 2*14 periods
        for i in range(28, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 1w ADX to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume filters
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or np.isnan(donchian_mid_4h[i]) or
            np.isnan(adx_4h[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 1d Donchian high with volume and strong trend
            if close[i] > donchian_high_4h[i] and volume_filter[i] and adx_4h[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below 1d Donchian low with volume and strong trend
            elif close[i] < donchian_low_4h[i] and volume_filter[i] and adx_4h[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint (mean reversion)
            if close[i] >= donchian_mid_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint (mean reversion)
            if close[i] <= donchian_mid_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals