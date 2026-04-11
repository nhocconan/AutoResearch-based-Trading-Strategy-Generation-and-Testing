#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with volume confirmation and ADX trend filter
# - Donchian(20) breakout: buy when price breaks above 20-period high, sell when breaks below 20-period low
# - Volume confirmation: current volume > 1.5x 20-period average volume
# - ADX trend filter: only trade when ADX > 25 (trending market) to avoid chop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - 1d ADX provides trend filter, reducing false signals in choppy markets

name = "12h_1d_donchian_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return signals
    
    # Pre-compute 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Plus Directional Movement (+DM)
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_plus[0] = 0
    
    # Minus Directional Movement (-DM)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_minus[0] = 0
    
    # Smoothed values (using Wilder's smoothing)
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (first 14 periods)
    atr[13] = np.mean(tr[1:15])
    dm_plus_smooth[13] = np.mean(dm_plus[1:15])
    dm_minus_smooth[13] = np.mean(dm_minus[1:15])
    
    # Wilder's smoothing for remaining periods
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / atr
    minus_di = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.zeros_like(atr)
    dx[13:] = 100 * np.abs(plus_di[13:] - minus_di[13:]) / (plus_di[13:] + minus_di[13:])
    
    adx = np.zeros_like(dx)
    adx[26] = np.mean(dx[14:28])  # First ADX value
    for i in range(27, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute Donchian channels on 12h data (20-period)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 12h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup for Donchian
        # Skip if any required data is invalid
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = price_close > high_roll_max[i-1]  # Close above previous period's high
        breakdown_down = price_close < low_roll_min[i-1]  # Close below previous period's low
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        trend_filter = adx_aligned[i] > 25
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout up + volume confirmation + trend filter
        if breakout_up and vol_confirm and trend_filter:
            enter_long = True
        
        # Short: Donchian breakdown down + volume confirmation + trend filter
        if breakdown_down and vol_confirm and trend_filter:
            enter_short = True
        
        # Exit conditions: opposite Donchian breakout or loss of trend
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if breakdown down OR ADX falls below 20 (loss of trend)
            exit_long = breakdown_down or (adx_aligned[i] < 20)
        elif position == -1:
            # Exit short if breakout up OR ADX falls below 20 (loss of trend)
            exit_short = breakout_up or (adx_aligned[i] < 20)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals