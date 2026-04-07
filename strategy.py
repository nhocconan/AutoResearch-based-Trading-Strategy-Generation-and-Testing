#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Weekly Donchian Breakout with Volume and ADX Filter
# Hypothesis: Price breaking out of weekly Donchian channels (20-period high/low)
# with volume confirmation and ADX trend filter (>25) captures strong momentum moves.
# ADX ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
# Volume > 1.5x 20-period average confirms institutional participation.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_weekly_donchian_breakout_volume_adx_v1"
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
    
    # Get weekly data for Donchian channels and ADX
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_high_series = pd.Series(weekly_high)
    weekly_low_series = pd.Series(weekly_low)
    donchian_high = weekly_high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = weekly_low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed weekly bars (avoid look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    
    # Handle first element
    if len(donchian_high) > 1:
        donchian_high[0] = donchian_high[1]
        donchian_low[0] = donchian_low[1]
    else:
        donchian_high[0] = 0
        donchian_low[0] = 0
    
    # Calculate weekly ADX (14-period)
    # ADX requires +DI, -DI, and TR
    tr1 = np.abs(np.diff(weekly_high))
    tr2 = np.abs(np.diff(weekly_low))
    tr3 = np.abs(weekly_high[:-1] - weekly_low[1:])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(np.maximum(tr1, tr2), tr3)])
    
    # Directional movement
    up_move = np.diff(weekly_high)
    down_move = -np.diff(weekly_low)  # positive when low decreases
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smooth = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
    minus_di = 100 * minus_dm_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = wilders_smooth(dx, 14)
    
    # Handle initial values
    if len(adx) > 1:
        adx[0] = adx[1]
    else:
        adx[0] = 0
    
    # Align weekly data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below weekly Donchian low or trend weakens (ADX < 20)
            if close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above weekly Donchian high or trend weakens (ADX < 20)
            if close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above weekly Donchian high with volume and strong trend (ADX > 25)
            if (high[i] > donchian_high_aligned[i] and close[i] > donchian_high_aligned[i] and
                vol_filter[i] and adx_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below weekly Donchian low with volume and strong trend (ADX > 25)
            elif (low[i] < donchian_low_aligned[i] and close[i] < donchian_low_aligned[i] and
                  vol_filter[i] and adx_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals