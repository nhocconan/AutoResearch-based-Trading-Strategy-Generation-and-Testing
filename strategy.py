#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation
# - Long when price breaks above 20-period high AND 1d ADX > 25 AND volume > 1.5x 20-bar avg
# - Short when price breaks below 20-period low AND 1d ADX > 25 AND volume > 1.5x 20-bar avg
# - Exit when price crosses the 20-period midpoint (mean reversion to equilibrium)
# - Uses 1d ADX for trend filter to avoid whipsaw in ranging markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years)
# - Donchian breakouts capture strong moves; ADX filter ensures we only trade in trending regimes

name = "6h_1d_donchian_breakout_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute Donchian channels from 6h data (20-period)
    high_6h = prices['high'].rolling(window=20, min_periods=20).max().values
    low_6h = prices['low'].rolling(window=20, min_periods=20).min().values
    mid_6h = (high_6h + low_6h) / 2  # 20-period midpoint for exit
    
    # Pre-compute 1d ADX(14) for trend filter
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
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align HTF ADX to LTF
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(high_6h[i]) or np.isnan(low_6h[i]) or np.isnan(mid_6h[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above 20-period high AND 1d ADX > 25 with volume spike
            if (prices['close'].iloc[i] > high_6h[i] and 
                adx_aligned[i] > 25 and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below 20-period low AND 1d ADX > 25 with volume spike
            elif (prices['close'].iloc[i] < low_6h[i] and 
                  adx_aligned[i] > 25 and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to midpoint (mean reversion)
            # Exit when price crosses the 20-period midpoint
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= mid_6h[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= mid_6h[i]:
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