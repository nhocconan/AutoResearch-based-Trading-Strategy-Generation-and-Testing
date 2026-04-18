#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with Elder Ray power and daily trend filter.
# Williams Alligator identifies trend direction using SMAs (Jaw/Teeth/Lips).
# Elder Ray measures bull/bear power relative to EMA13 to confirm trend strength.
# Daily EMA50 filter ensures alignment with higher-timeframe trend to avoid counter-trend trades.
# Designed for low trade frequency (15-35/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (bull power > 0 with price above teeth) and bear markets (bear power < 0 with price below teeth).
name = "6h_WilliamsAlligator_ElderRay_DailyEMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for EMA50 trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator components (13,8,5 SMAs shifted)
    # Jaw: 13-period SMA shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips above Teeth (bullish alignment) AND bull power > 0 AND price above daily EMA50
            if lips[i] > teeth[i] and bull_power[i] > 0 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips below Teeth (bearish alignment) AND bear power < 0 AND price below daily EMA50
            elif lips[i] < teeth[i] and bear_power[i] < 0 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Lips cross below Teeth OR bear power becomes negative
            if lips[i] < teeth[i] or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Lips cross above Teeth OR bull power becomes positive
            if lips[i] > teeth[i] or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals