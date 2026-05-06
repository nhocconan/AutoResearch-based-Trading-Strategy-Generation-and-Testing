#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA(50) trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Trend filter: 1d EMA(50) slope > 0 for long, < 0 for short
# Volume: > 1.5x 20-bar average confirms participation
# Works in bull/bear: follows trend with momentum confirmation, avoids whipsaws in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_ElderRay_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 55:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope_1d = np.diff(ema_50_1d, prepend=ema_50_1d[0])
    
    # Calculate EMA(13) for Elder Ray (6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume filter: > 1.5x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    ema_50_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_slope_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema_50_slope_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, 1d EMA50 slope > 0 (uptrend), volume confirmation
            if bull_power[i] > 0 and ema_50_slope_1d_aligned[i] > 0 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, 1d EMA50 slope < 0 (downtrend), volume confirmation
            elif bear_power[i] < 0 and ema_50_slope_1d_aligned[i] < 0 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR 1d EMA50 slope <= 0 (trend weakening)
            if bull_power[i] <= 0 or ema_50_slope_1d_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR 1d EMA50 slope >= 0 (trend weakening)
            if bear_power[i] >= 0 or ema_50_slope_1d_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals