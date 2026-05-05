#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper AND 1w close > 1w EMA34 AND volume > 2.0x 20 EMA
# Short when price breaks below 1d Donchian lower AND 1w close < 1w EMA34 AND volume > 2.0x 20 EMA
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-30 trades/year per symbol.
# Donchian channels provide strong trend-following structure; 1w EMA34 filters counter-trend trades;
# Volume spike confirms institutional participation. Works in bull markets via longs and bear via shorts.

name = "1d_Donchian20_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation (primary timeframe is 1d, so we use prices directly)
    # But we still need to ensure we use completed daily bars only via proper alignment
    # Since prices is already 1d, we can calculate Donchian directly
    if n < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    # We use rolling window with min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous bar's levels for breakout (avoid look-ahead)
    donchian_upper = np.roll(donchian_upper, 1)
    donchian_lower = np.roll(donchian_lower, 1)
    donchian_upper[0] = np.nan
    donchian_lower[0] = np.nan
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1w = close_1w > ema_34_1w
    downtrend_1w = close_1w < ema_34_1w
    
    # Align 1w trend to 1d timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian upper AND 1w uptrend AND volume spike
            if (close[i] > donchian_upper[i] and 
                uptrend_1w_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian lower AND 1w downtrend AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  downtrend_1w_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian lower OR 1w trend changes to downtrend
            if (close[i] < donchian_lower[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian upper OR 1w trend changes to uptrend
            if (close[i] > donchian_upper[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals