#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA trend filter and volume confirmation
# - Long when price breaks above 20-period Donchian upper channel AND 12h HMA(21) rising AND volume > 1.5x 20-period average volume
# - Short when price breaks below 20-period Donchian lower channel AND 12h HMA(21) falling AND volume > 1.5x 20-period average volume
# - Exit when price crosses back inside the Donchian channel (between upper and lower bands)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Donchian channels identify clear breakouts with defined risk levels
# - HMA trend filter ensures we trade in the direction of the higher timeframe trend
# - Volume confirmation reduces false breakouts

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    upper_channel = rolling_max(high, 20)
    lower_channel = rolling_min(low, 20)
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 12h HMA(21) for trend filter
    def wma(data, window):
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights / weights.sum(), mode='valid')
    
    def hma(data, window):
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        wma_half = wma(data, half_window)
        wma_full = wma(data, window)
        wma_half_2x = 2 * wma_half
        diff = wma_half_2x - wma_full
        return wma(diff, sqrt_window)
    
    close_12h = df_12h['close'].values
    hma_12h = np.full_like(close_12h, np.nan, dtype=float)
    if len(close_12h) >= 21:
        hma_values = hma(close_12h, 21)
        start_idx = 21 - 1  # HMA(21) needs 21 periods
        hma_12h[start_idx:start_idx + len(hma_values)] = hma_values
    
    # HMA trend: rising when current > previous, falling when current < previous
    hma_rising = np.full_like(hma_12h, False, dtype=bool)
    hma_falling = np.full_like(hma_12h, False, dtype=bool)
    for i in range(1, len(hma_12h)):
        if not np.isnan(hma_12h[i]) and not np.isnan(hma_12h[i-1]):
            if hma_12h[i] > hma_12h[i-1]:
                hma_rising[i] = True
            elif hma_12h[i] < hma_12h[i-1]:
                hma_falling[i] = True
    
    # Align HTF indicators to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_12h, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_12h, hma_falling)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above upper channel AND HMA rising AND volume spike
            if (close[i] > upper_channel[i] and 
                hma_rising_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below lower channel AND HMA falling AND volume spike
            elif (close[i] < lower_channel[i] and 
                  hma_falling_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside the Donchian channel
            exit_long = (position == 1 and close[i] < upper_channel[i])
            exit_short = (position == -1 and close[i] > lower_channel[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals