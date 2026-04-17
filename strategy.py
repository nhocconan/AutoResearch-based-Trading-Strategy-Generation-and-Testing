#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
# Enters long when price breaks above 4h Donchian upper band (20) AND 12h EMA(50) is rising AND volume > 1.5x 20-period average.
# Enters short when price breaks below 4h Donchian lower band (20) AND 12h EMA(50) is falling AND volume > 1.5x 20-period average.
# Exits when price crosses the 4h Donchian middle line (20-period average of high/low).
# Trend filter ensures we only trade in the direction of the 12h trend, reducing whipsaws.
# Volume confirmation adds conviction. Designed for low turnover (target: 20-50 trades/year).
# Works in bull markets (trend continuation up) and bear markets (trend continuation down).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 4h Donchian Channel (20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 12h EMA(50)
    close_12h_series = pd.Series(close_12h)
    ema_12h_50 = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_50_4h = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Calculate 12h EMA(50) slope (rising/falling)
    ema_slope = np.zeros_like(ema_12h_50_4h)
    ema_slope[1:] = ema_12h_50_4h[1:] - ema_12h_50_4h[:-1]
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_12h_50_4h[i]) or 
            np.isnan(ema_slope[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1] if i > 0 else False
        breakout_down = close[i] < donchian_low[i-1] if i > 0 else False
        
        # EMA trend filter: rising or falling
        ema_rising = ema_slope[i] > 0
        ema_falling = ema_slope[i] < 0
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: Donchian breakout up + EMA rising + volume
            if breakout_up and ema_rising and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + EMA falling + volume
            elif breakout_down and ema_falling and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian middle
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian middle
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0