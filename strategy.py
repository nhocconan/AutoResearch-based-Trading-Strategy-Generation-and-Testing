#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d volume confirmation and 1w ADX trend filter
# - Long when price breaks above 12h Donchian upper (20) AND 1d volume > 1.8x 20-bar avg AND 1w ADX > 25 (trending market)
# - Short when price breaks below 12h Donchian lower (20) AND 1d volume > 1.8x 20-bar avg AND 1w ADX > 25 (trending market)
# - Exit when price returns to 12h Donchian middle (median of upper/lower) or opposite band touch
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Donchian breakouts capture strong momentum; volume confirms institutional participation
# - Weekly ADX filter ensures we only trade in trending regimes, reducing whipsaws in ranging markets

name = "12h_1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Calculate Donchian channels: upper = max(high, lookback), lower = min(low, lookback)
    lookback = 20
    donchian_upper = pd.Series(high_12h).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=lookback, min_periods=lookback).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Pre-compute 1d volume confirmation: > 1.8x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 1w ADX trend filter (ADX > 25 = trending market)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First value has no previous close
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (simple average)
    atr[period] = np.mean(tr[1:period+1])
    dm_plus_smooth[period] = np.mean(dm_plus[1:period+1])
    dm_minus_smooth[period] = np.mean(dm_minus[1:period+1])
    
    # Wilder's smoothing
    for i in range(period + 1, len(tr)):
        atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
        dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / period) + dm_plus[i]
        dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / period) + dm_minus[i]
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros_like(dx)
    adx[2*period] = np.mean(dx[period:2*period+1])  # Initial ADX value
    
    for i in range(2*period + 1, len(dx)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # ADX > 25 indicates trending market
    adx_trending = adx > 25.0
    adx_trending_aligned = align_htf_to_ltf(prices, df_1w, adx_trending)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(adx_trending_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper AND 1d volume spike AND weekly trending
            if (prices['close'].iloc[i] > donchian_upper[i] and 
                vol_spike_1d_aligned[i] and 
                adx_trending_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND 1d volume spike AND weekly trending
            elif (prices['close'].iloc[i] < donchian_lower[i] and 
                  vol_spike_1d_aligned[i] and 
                  adx_trending_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to Donchian middle OR touches opposite band (reversal signal)
            exit_signal = False
            if position == 1:  # Long position
                if (prices['close'].iloc[i] <= donchian_middle[i] or 
                    prices['close'].iloc[i] >= donchian_upper[i]):  # Touch upper band = potential exhaustion
                    exit_signal = True
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] >= donchian_middle[i] or 
                    prices['close'].iloc[i] <= donchian_lower[i]):  # Touch lower band = potential exhaustion
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals