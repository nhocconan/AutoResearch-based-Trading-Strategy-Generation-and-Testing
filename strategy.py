#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h trend following with 1w EMA filter and volume spike confirmation.
    # Long when price > 6h EMA(50) AND 1w EMA(34) rising AND volume > 1.5x 20-period average.
    # Short when price < 6h EMA(50) AND 1w EMA(34) falling AND volume > 1.5x 20-period average.
    # Uses 6h EMA for trend, 1w EMA for higher-timeframe bias, volume spike for momentum confirmation.
    # Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    # Calculate 1w EMA slope (rising/falling) - needs 2-bar lookback for confirmation
    ema_34_1w_slope = np.diff(ema_34_1w_aligned, prepend=ema_34_1w_aligned[0])
    ema_34_1w_rising = ema_34_1w_slope > 0
    ema_34_1w_falling = ema_34_1w_slope < 0
    
    # Calculate 6h EMA(50) for trend
    ema_50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # Discrete level to minimize fee churn
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_6h[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_setup = (close[i] > ema_50_6h[i]) and ema_34_1w_rising[i] and volume_spike[i]
        short_setup = (close[i] < ema_50_6h[i]) and ema_34_1w_falling[i] and volume_spike[i]
        
        # Exit conditions: reverse signal or volume dry-up
        long_exit = (close[i] < ema_50_6h[i]) or not volume_spike[i]
        short_exit = (close[i] > ema_50_6h[i]) or not volume_spike[i]
        
        # Entry logic
        if long_setup and position != 1:
            position = 1
            signals[i] = position_size
        elif short_setup and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit logic
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_ema_trend_volume_spike_v1"
timeframe = "6h"
leverage = 1.0