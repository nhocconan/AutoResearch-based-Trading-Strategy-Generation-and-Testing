#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h volume confirmation and 1d ADX regime filter
# - Long when price breaks above Donchian(20) high AND 12h volume > 1.5x 20-bar avg AND 1d ADX < 25 (range/low trend)
# - Short when price breaks below Donchian(20) low AND 12h volume > 1.5x 20-bar avg AND 1d ADX < 25
# - Exit when price touches Donchian midpoint (mean reversion within channel)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian breakouts capture momentum; volume confirmation validates breakout strength
# - ADX < 25 filter avoids false breakouts in strong trends where price may continue beyond channel
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in both bull and bear markets: breakouts work in trends, mean reversion exit works in ranges

name = "4h_12h_1d_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume_12h = df_12h['volume'].values
    volume_20_avg = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.5 * volume_20_avg)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR and DM
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ADX < 25 indicates weak trend (range/low trend environment)
    adx_low = adx < 25
    adx_low_aligned = align_htf_to_ltf(prices, df_1d, adx_low)
    
    # Pre-compute Donchian Channel(20) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Donchian breakout conditions
    breakout_up = close > highest_high
    breakout_down = close < lowest_low
    
    # Exit when price touches Donchian midpoint (mean reversion)
    midpoint_exit = np.abs(close - donchian_mid) < (highest_high - lowest_low) * 0.05
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(vol_spike_12h_aligned[i]) or np.isnan(adx_low_aligned[i]) or
            np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(midpoint_exit[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when bullish breakout AND volume spike AND low ADX (range/low trend)
            if (breakout_up[i] and 
                vol_spike_12h_aligned[i] and 
                adx_low_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when bearish breakout AND volume spike AND low ADX
            elif (breakout_down[i] and 
                  vol_spike_12h_aligned[i] and 
                  adx_low_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Donchian midpoint
            # Exit when price returns to Donchian midpoint (mean reversion)
            exit_signal = midpoint_exit[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals