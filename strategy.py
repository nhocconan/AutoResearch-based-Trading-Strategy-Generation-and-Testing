#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Bollinger Band mean reversion with 1w ADX trend filter
# Long when price touches lower BB(20,2) on 1d AND 1w ADX < 25 (ranging market) AND volume > 1.5 * avg_volume(20) on 6h
# Short when price touches upper BB(20,2) on 1d AND 1w ADX < 25 AND volume spike
# Exit when price crosses back through BB middle (20-period SMA) on 1d
# Uses discrete sizing 0.25 to limit fee churn
# Target: 80-180 total trades over 4 years (20-45/year) for 6h timeframe
# Bollinger Band touches provide high-probability mean reversion entries in ranging markets
# 1w ADX < 25 filter ensures we only trade when market is ranging (not trending)
# Volume confirmation validates the touch while avoiding low-liquidity false signals
# Works in both bull (buy dips in range) and bear (sell rallies in range) markets

name = "6h_1dBB_Touch_1wADX_Range_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed 1d bars for BB
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Bollinger Bands: SMA(20) ± 2*STD(20)
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2.0 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2.0 * std_20_1d)
    middle_bb_1d = sma_20_1d  # 20-period SMA
    
    # Align 1d Bollinger Bands to 6h timeframe (wait for completed 1d bar)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    middle_bb_aligned = align_htf_to_ltf(prices, df_1d, middle_bb_1d)
    
    # Get 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need at least 30 completed weekly bars for ADX(14)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14): Average Directional Index
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w != 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = wilders_smoothing(dx, 14)
    
    # Align 1w ADX to 6h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(lower_bb_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches lower BB, 1w ADX < 25 (ranging), volume spike, in session
            if (low[i] <= lower_bb_aligned[i] and 
                adx_1w_aligned[i] < 25 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB, 1w ADX < 25, volume spike, in session
            elif (high[i] >= upper_bb_aligned[i] and 
                  adx_1w_aligned[i] < 25 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back above middle BB (mean reversion)
            if close[i] >= middle_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back below middle BB (mean reversion)
            if close[i] <= middle_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals