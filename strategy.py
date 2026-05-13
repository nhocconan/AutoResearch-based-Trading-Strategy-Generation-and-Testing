#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (EMA50) and volume confirmation.
# Long when price breaks above Donchian upper (20-bar high) AND 12h EMA50 rising AND volume > 1.5x average.
# Short when price breaks below Donchian lower (20-bar low) AND 12h EMA50 falling AND volume > 1.5x average.
# Exit when price touches Donchian midpoint (mean of upper/lower) OR trend reversal (EMA50 slope changes).
# Uses 4h timeframe for optimal trade frequency (target: 75-200 trades over 4 years).
# Donchian provides clear structure, 12h EMA50 filters trend direction, volume confirms breakout strength.
# Works in bull via continuation breakouts, bear via faded rallies and mean reversion to midpoint.

name = "4h_Donchian20_12hEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian upper (20-bar high) and lower (20-bar low) on 4h
    donch_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_upper + donch_lower) / 2.0
    
    # Align Donchian levels to 15m timeframe (assuming 4h = 16x 15m, but using helper)
    donch_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_lower)
    donch_mid_aligned = align_htf_to_ltf(prices, df_4h, donch_mid)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate EMA50 slope (rising/falling) on 12h - need 2 values to compute slope
    ema50_12h_prev = np.roll(ema50_12h_aligned, 1)
    ema50_12h_prev[0] = ema50_12h_aligned[0]  # handle first value
    ema50_rising = ema50_12h_aligned > ema50_12h_prev
    ema50_falling = ema50_12h_aligned < ema50_12h_prev
    
    # Volume filter: current 4h volume > 1.5x 20-period average
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or
            np.isnan(donch_mid_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper AND 12h EMA50 rising AND volume confirmation
            if close[i] > donch_upper_aligned[i] and ema50_rising[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower AND 12h EMA50 falling AND volume confirmation
            elif close[i] < donch_lower_aligned[i] and ema50_falling[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches Donchian midpoint OR 12h EMA50 starts falling (trend reversal)
            if close[i] <= donch_mid_aligned[i] or not ema50_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches Donchian midpoint OR 12h EMA50 starts rising (trend reversal)
            if close[i] >= donch_mid_aligned[i] or not ema50_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals