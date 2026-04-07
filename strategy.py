#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1D Weekly Donchian Breakout with Volume and ADX Filter
# Hypothesis: Weekly trend + daily breakout provides strong directional moves.
# Long when price breaks above weekly Donchian high with volume and ADX>25.
# Short when price breaks below weekly Donchian low with volume and ADX>25.
# Works in bull markets (buy breakouts) and bear markets (sell breakdowns).
# Weekly timeframe captures major trends, daily provides entry timing.
# Target: 7-25 trades/year (30-100 total over 4 years).

name = "1d_weekly_donchian_breakout_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly high/low for Donchian(20)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_series = pd.Series(weekly_high)
    low_series = pd.Series(weekly_low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe (shifted by 1 for completed bars)
    dh_align = align_htf_to_ltf(prices, df_weekly, donchian_high)
    dl_align = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ADX filter: ADX > 25 for trending markets
    # Calculate ADX using Wilder's smoothing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(tr)
    minus_di = np.zeros_like(tr)
    dx = np.zeros_like(tr)
    
    atr[0] = tr[0]
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    period = 14
    alpha = 1.0 / period
    
    for i in range(1, n):
        # Wilder smoothing
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
        plus_dm_smooth[i] = (1 - alpha) * plus_dm_smooth[i-1] + alpha * plus_dm[i]
        minus_dm_smooth[i] = (1 - alpha) * minus_dm_smooth[i-1] + alpha * minus_dm[i]
        
        if atr[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is smoothed DX
    adx = np.zeros_like(dx)
    adx[period] = dx[period]  # First ADX is DX at period
    for i in range(period + 1, n):
        adx[i] = (1 - alpha) * adx[i-1] + alpha * dx[i]
    
    adx_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(dh_align[i]) or np.isnan(dl_align[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below weekly Donchian low
            if close[i] < dl_align[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above weekly Donchian high
            if close[i] > dh_align[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above weekly Donchian high with volume and ADX
            if (high[i] > dh_align[i] and close[i] > dh_align[i] and
                vol_filter[i] and adx_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below weekly Donchian low with volume and ADX
            elif (low[i] < dl_align[i] and close[i] < dl_align[i] and
                  vol_filter[i] and adx_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals