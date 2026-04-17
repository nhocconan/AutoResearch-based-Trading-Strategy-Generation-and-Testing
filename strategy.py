#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h EMA34 trend filter + 1d Donchian20 breakout + volume confirmation.
Long when price breaks above 1d Donchian upper (20-period high) with 12h EMA34 > prior 12h EMA34 and volume > 1.5x 20-period 1d volume average.
Short when price breaks below 1d Donchian lower (20-period low) with 12h EMA34 < prior 12h EMA34 and volume > 1.5x 20-period 1d volume average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
Volume confirmation ensures breakouts have participation, reducing false signals.
Works in bull markets (trend continuation) and bear markets (mean reversion after volatility expansion).
"""

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
    
    # Get 12h data for EMA34 trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Get 1d data for Donchian levels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_prev = np.roll(ema_34_12h, 1)
    ema_34_12h_prev[0] = np.nan
    ema_34_12h_rising = ema_34_12h > ema_34_12h_prev
    ema_34_12h_falling = ema_34_12h < ema_34_12h_prev
    
    # Calculate 1d Donchian channels (20-period)
    donch_hi_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_lo_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    ema_34_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_rising)
    ema_34_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_falling)
    donch_hi_20_aligned = align_htf_to_ltf(prices, df_1d, donch_hi_20)
    donch_lo_20_aligned = align_htf_to_ltf(prices, df_1d, donch_lo_20)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_hi_20_aligned[i]) or np.isnan(donch_lo_20_aligned[i]) or 
            np.isnan(ema_34_12h_rising_aligned[i]) or np.isnan(ema_34_12h_falling_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian high with rising 12h EMA34 and volume
            if (close[i] > donch_hi_20_aligned[i] and 
                ema_34_12h_rising_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low with falling 12h EMA34 and volume
            elif (close[i] < donch_lo_20_aligned[i] and 
                  ema_34_12h_falling_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 1d Donchian midpoint
            midpoint = (donch_hi_20 + donch_lo_20) / 2
            midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint)
            if close[i] < midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 1d Donchian midpoint
            midpoint = (donch_hi_20 + donch_lo_20) / 2
            midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint)
            if close[i] > midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hEMA34_1dDonchian20_Volume_Confirm"
timeframe = "4h"
leverage = 1.0