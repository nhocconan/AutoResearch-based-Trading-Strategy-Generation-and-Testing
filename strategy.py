#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Williams %R extreme reversal + 12h EMA34 trend filter + volume spike confirmation.
Long when 1d Williams %R crosses above -80 from oversold with rising 12h EMA34 and volume > 1.5x 20-period average.
Short when 1d Williams %R crosses below -20 from overbought with falling 12h EMA34 and volume > 1.5x 20-period average.
Williams %R captures mean reversion in bear markets, EMA34 filters trend alignment, volume spike confirms participation.
Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag in ranging/bear markets.
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
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 12h data for EMA34 trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    
    # Calculate Williams %R cross above -80 (oversold) and below -20 (overbought)
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    williams_cross_above_80 = (williams_r > -80) & (williams_r_prev <= -80)
    williams_cross_below_20 = (williams_r < -20) & (williams_r_prev >= -20)
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_prev = np.roll(ema_34_12h, 1)
    ema_34_12h_prev[0] = np.nan
    ema_34_12h_rising = ema_34_12h > ema_34_12h_prev
    ema_34_12h_falling = ema_34_12h < ema_34_12h_prev
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    williams_cross_above_80_aligned = align_htf_to_ltf(prices, df_1d, williams_cross_above_80.astype(float))
    williams_cross_below_20_aligned = align_htf_to_ltf(prices, df_1d, williams_cross_below_20.astype(float))
    ema_34_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_rising.astype(float))
    ema_34_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_falling.astype(float))
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(34, 20, 14)  # need enough for EMA34, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_cross_above_80_aligned[i]) or np.isnan(williams_cross_below_20_aligned[i]) or 
            np.isnan(ema_34_12h_rising_aligned[i]) or np.isnan(ema_34_12h_falling_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from oversold with rising 12h EMA34 and volume spike
            if (williams_cross_above_80_aligned[i] > 0.5 and 
                ema_34_12h_rising_aligned[i] > 0.5 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from overbought with falling 12h EMA34 and volume spike
            elif (williams_cross_below_20_aligned[i] > 0.5 and 
                  ema_34_12h_falling_aligned[i] > 0.5 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) or 12h EMA34 starts falling
            if (williams_cross_below_20_aligned[i] > 0.5 or 
                ema_34_12h_falling_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) or 12h EMA34 starts rising
            if (williams_cross_above_80_aligned[i] > 0.5 or 
                ema_34_12h_rising_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dWilliamsR_12hEMA34_Volume_Spike"
timeframe = "6h"
leverage = 1.0