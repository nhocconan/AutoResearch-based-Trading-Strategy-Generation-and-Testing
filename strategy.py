#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend filter + volume confirmation
# - Long when price breaks above Donchian(20) high AND 1w HMA(21) is rising AND volume > 1.5x 20-period average
# - Short when price breaks below Donchian(20) low AND 1w HMA(21) is falling AND volume > 1.5x 20-period average
# - Exit when price crosses Donchian(20) midline
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Donchian breakouts capture strong momentum moves in both bull and bear markets
# - 1w HMA filter ensures we trade with the higher timeframe trend, reducing whipsaw
# - Volume confirmation reduces false breakouts

name = "1d_1w_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Pre-compute 1d Donchian channels (20)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian high/low (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Pre-compute 1d volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w HMA(21) for trend filter
    close_1w = df_1w['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, n):
        if len(arr) < n:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, n + 1)
        return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
    
    # Calculate WMA for n/2 and n
    n = 21
    half_n = n // 2
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_1w, half_n)
    wma_full = wma(close_1w, n)
    
    # Handle array lengths
    wma_half_padded = np.full_like(close_1w, np.nan)
    wma_full_padded = np.full_like(close_1w, np.nan)
    
    if len(wma_half) > 0:
        wma_half_padded[half_n-1:half_n-1+len(wma_half)] = wma_half
    if len(wma_full) > 0:
        wma_full_padded[n-1:n-1+len(wma_full)] = wma_full
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half_padded - wma_full_padded
    # WMA(sqrt(n)) of the diff
    wma_diff = wma(diff, sqrt_n)
    wma_diff_padded = np.full_like(close_1w, np.nan)
    if len(wma_diff) > 0:
        wma_diff_padded[sqrt_n-1:sqrt_n-1+len(wma_diff)] = wma_diff
    
    hma_1w = wma_diff_padded
    
    # HMA slope (rising/falling)
    hma_slope = np.diff(hma_1w, prepend=np.nan)
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    # Align HTF indicators to 1d timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(hma_rising_aligned[i]) or 
            np.isnan(hma_falling_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND 1w HMA rising AND volume spike
            if (close[i] > donch_high[i] and 
                hma_rising_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND 1w HMA falling AND volume spike
            elif (close[i] < donch_low[i] and 
                  hma_falling_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Donchian midline
            exit_long = (position == 1 and close[i] < donch_mid[i])
            exit_short = (position == -1 and close[i] > donch_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals