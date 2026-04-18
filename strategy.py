#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(34) trend filter and volume confirmation.
# Long when price breaks above Donchian upper (20-period high) and 12h EMA(34) > previous.
# Short when price breaks below Donchian lower (20-period low) and 12h EMA(34) < previous.
# Volume must be > 1.5x 20-period average to confirm breakout strength.
# Exit when price crosses the opposite Donchian band or EMA trend reverses.
# Designed for 4h timeframe to target 75-200 trades over 4 years (19-50/year).
# Works in bull markets (breakouts up) and bear markets (breakouts down).
# Uses tight entry conditions to minimize fee drag and avoid overtrading.

name = "4h_Donchian20_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for EMA(34) trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(34) on 12h close
    close_12h = pd.Series(df_12h['close'].values)
    ema_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND EMA rising AND volume confirmed
            if (close[i] > donchian_high[i] and 
                ema_12h_aligned[i] > ema_12h_aligned[i-1] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND EMA falling AND volume confirmed
            elif (close[i] < donchian_low[i] and 
                  ema_12h_aligned[i] < ema_12h_aligned[i-1] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian low OR EMA turns down
            if (close[i] < donchian_low[i] or 
                ema_12h_aligned[i] < ema_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian high OR EMA turns up
            if (close[i] > donchian_high[i] or 
                ema_12h_aligned[i] > ema_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals