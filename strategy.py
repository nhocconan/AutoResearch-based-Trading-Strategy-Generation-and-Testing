#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 12h trend filter (EMA50 slope) and 1d volume spike confirmation.
# Long when price breaks above Donchian upper AND 12h EMA50 slope > 0 (uptrend) AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below Donchian lower AND 12h EMA50 slope < 0 (downtrend) AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel (mean of upper and lower).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 6h timeframe with strict entry conditions.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h.
# Donchian structure provides clear support/resistance, EMA50 slope filters choppy markets, volume spike confirms institutional interest.

name = "6h_Donchian20_Breakout_12hEMA50Slope_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 slope for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 55:  # Need enough for EMA50
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # EMA50 calculation
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Slope of EMA50 (change over 2 periods to smooth noise)
    ema50_slope = np.full_like(ema_50, np.nan)
    ema50_slope[2:] = (ema_50[2:] - ema_50[:-2]) / 2  # 2-period slope
    ema50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema50_slope)
    
    # Calculate 1d volume confirmation filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:  # Need enough for volume MA
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Donchian channel (20-period) on primary timeframe
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    # Highest high and lowest low over past 20 periods (including current)
    for i in range(20-1, n):  # Start from index 19 (20th bar)
        donchian_upper[i] = np.max(high[i-20+1:i+1])
        donchian_lower[i] = np.min(low[i-20+1:i+1])
        donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-compute session hours to avoid datetime operations in loop
    session_hours = prices.index.hour
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(ema50_slope_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC (reduce off-hour noise)
        hour = session_hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper AND 12h EMA50 slope > 0 (uptrend) AND volume confirmation
            if (open_[i] <= donchian_upper[i] and close[i] > donchian_upper[i] and 
                ema50_slope_aligned[i] > 0 and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower AND 12h EMA50 slope < 0 (downtrend) AND volume confirmation
            elif (open_[i] >= donchian_lower[i] and close[i] < donchian_lower[i] and 
                  ema50_slope_aligned[i] < 0 and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals