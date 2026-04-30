#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses Donchian channel from 1d for structure (proven edge on multiple symbols)
# Only trade breakouts above upper band or below lower band in direction of 1w EMA34 trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# 1w EMA34 provides smoother trend than shorter EMAs, reducing whipsaw in ranging markets
# Discrete sizing 0.25 minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).
# Works in both bull and bear markets by following the 1w EMA34 trend direction.

name = "1d_Donchian20_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Donchian(20) - using prior completed 1d bar to avoid look-ahead
    # We need the last 20 completed 1d bars, so we shift by 1
    high_1d = df_1w['high'].values  # Using 1w data for Donchian would be wrong, need 1d
    # Correction: we need 1d data for Donchian calculation
    # But we already have prices which is 1d TF, so we can use it directly with proper shifting
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # We need to calculate Donchian on 1d data, but we must use prior completed bars
    # So we'll calculate rolling max/min on shifted series
    
    # For Donchian(20), we need the highest high and lowest low of the prior 20 completed 1d bars
    # Since we're on 1d timeframe, we can use prices directly but shift by 1 to avoid look-ahead
    
    # Pre-calculate Donchian levels using prior 20 bars (shifted by 1)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    # Shift by 1 to use only prior completed bars
    high_shifted = high_series.shift(1)
    low_shifted = low_series.shift(1)
    # Rolling window of 20 on the shifted series
    donchian_high = high_shifted.rolling(window=20, min_periods=20).max().values
    donchian_low = low_shifted.rolling(window=20, min_periods=20).min().values
    
    start_idx = 20  # warmup for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if Donchian levels are not available (NaN)
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above upper band AND above 1w EMA34 (uptrend)
                if curr_close > upper_band and curr_close > curr_ema_34_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below lower band AND below 1w EMA34 (downtrend)
                elif curr_close < lower_band and curr_close < curr_ema_34_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below lower band or below 1w EMA34
            if curr_close < lower_band or curr_close < curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above upper band or above 1w EMA34
            if curr_close > upper_band or curr_close > curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals