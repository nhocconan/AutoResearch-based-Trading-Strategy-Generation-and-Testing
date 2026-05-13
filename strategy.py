#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h Donchian(20) breakout for signal direction and 1d EMA(50) for trend filter.
# Volume spike (>2x 20-period average) confirms breakout strength.
# Exit on Donchian middle cross or trend reversal (price crosses 1d EMA50).
# Uses 4h/1d for direction (lower trade frequency) and 1h for precise entry timing.
# Session filter (08-20 UTC) reduces noise trades. Fixed size 0.20 to control risk and fees.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.

name = "1h_Donchian20_1dEMA50_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC) - index is already DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian(20) on 4h data (using previous 20 bars)
    if len(high_4h) >= 20:
        upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
        lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
        middle_4h = (upper_4h + lower_4h) / 2
    else:
        upper_4h = np.full_like(high_4h, np.nan)
        lower_4h = np.full_like(low_4h, np.nan)
        middle_4h = np.full_like(high_4h, np.nan)
    
    # Align Donchian levels to 1h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    middle_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 1h volume > 2.0x 20-period average (spike confirmation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for Donchian and EMA
        # Skip if any required data is NaN or outside session
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > upper AND close > 1d EMA50 AND volume spike
            if close[i] > upper_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: price < lower AND close < 1d EMA50 AND volume spike
            elif close[i] < lower_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < middle (mean reversion) OR trend reversal (close < 1d EMA50)
            if close[i] < middle_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price > middle (mean reversion) OR trend reversal (close > 1d EMA50)
            if close[i] > middle_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals