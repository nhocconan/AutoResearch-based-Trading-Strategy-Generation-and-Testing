#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and volume confirmation.
    # Long when Bull Power > 0, 12h EMA rising, and volume > 1.5x 20-period mean.
    # Short when Bear Power < 0, 12h EMA falling, and volume > 1.5x 20-period mean.
    # Exit when Elder Power reverses or volume drops.
    # Uses discrete size 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_12h = close_12h_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_rising = ema_12h > np.roll(ema_12h, 1)  # EMA rising vs previous bar
    ema_12h_falling = ema_12h < np.roll(ema_12h, 1)  # EMA falling vs previous bar
    
    # Align 12h EMA and trend to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    ema_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_rising)
    ema_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_falling)
    
    # Calculate 12h volume mean (20-period) with min_periods
    volume_12h = df_12h['volume'].values
    volume_12h_series = pd.Series(volume_12h)
    vol_ma_20_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h EMA13)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13  # Bull Power: ability of bulls to push price above EMA
    bear_power = low - ema_13   # Bear Power: ability of bears to push price below EMA
    
    # Volume filter: current 12h volume > 1.5x 20-period mean (volume spike)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    volume_confirmation = volume_12h_aligned > 1.5 * vol_ma_20_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(ema_12h_rising_aligned[i]) or
            np.isnan(ema_12h_falling_aligned[i]) or np.isnan(volume_confirmation[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (bull_power[i] > 0 and ema_12h_rising_aligned[i] and volume_confirmation[i])
        short_entry = (bear_power[i] < 0 and ema_12h_falling_aligned[i] and volume_confirmation[i])
        
        # Exit conditions: Elder Power reverses or volume drops
        long_exit = (bull_power[i] <= 0) or (not volume_confirmation[i])
        short_exit = (bear_power[i] >= 0) or (not volume_confirmation[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_elder_ray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0