#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d HMA trend filter and volume confirmation
# - Long when price breaks above Camarilla H4 level AND 1d HMA(21) rising AND volume > 1.8x 20-period average
# - Short when price breaks below Camarilla L4 level AND 1d HMA(21) falling AND volume > 1.8x 20-period average
# - Exit when price returns to Camarilla Pivot point (midline) OR opposite breakout occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla levels from 1d provide strong intraday support/resistance
# - 1d HMA filter ensures alignment with daily trend
# - Volume confirmation reduces false breakouts
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_camarilla_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Pre-compute 1d HMA(21) for trend filter
    def wma(arr, n):
        if len(arr) < n:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, n + 1)
        return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
    
    close_1d = df_1d['close'].values
    n_hma = 21
    half_n = n_hma // 2
    sqrt_n = int(np.sqrt(n_hma))
    
    wma_half = wma(close_1d, half_n)
    wma_full = wma(close_1d, n_hma)
    
    wma_half_padded = np.full_like(close_1d, np.nan)
    wma_full_padded = np.full_like(close_1d, np.nan)
    
    if len(wma_half) > 0:
        wma_half_padded[half_n-1:half_n-1+len(wma_half)] = wma_half
    if len(wma_full) > 0:
        wma_full_padded[n_hma-1:n_hma-1+len(wma_full)] = wma_full
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half_padded - wma_full_padded
    # WMA(sqrt(n)) of the diff
    wma_diff = wma(diff, sqrt_n)
    wma_diff_padded = np.full_like(close_1d, np.nan)
    if len(wma_diff) > 0:
        wma_diff_padded[sqrt_n-1:sqrt_n-1+len(wma_diff)] = wma_diff
    
    hma_1d = wma_diff_padded
    hma_slope = np.diff(hma_1d, prepend=np.nan)
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    # Align HTF indicators to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Camarilla levels using prior 1d OHLC (aligned to 4h)
    # We need prior day's OHLC for each 4h bar
    df_1d_index = df_1d.index
    # Create arrays for prior 1day OHLC aligned to each 4h bar
    pri_open = align_htf_to_ltf(prices, df_1d, df_1d['open'].values, additional_delay_bars=1)
    pri_high = align_htf_to_ltf(prices, df_1d, df_1d['high'].values, additional_delay_bars=1)
    pri_low = align_htf_to_ltf(prices, df_1d, df_1d['low'].values, additional_delay_bars=1)
    pri_close = align_htf_to_ltf(prices, df_1d, df_1d['close'].values, additional_delay_bars=1)
    
    # Camarilla levels calculation
    # Pivot = (pri_high + pri_low + pri_close) / 3
    pivot = (pri_high + pri_low + pri_close) / 3
    range_hl = pri_high - pri_low
    
    # Resistance levels
    r4 = pri_close + range_hl * 1.500  # H4
    r3 = pri_close + range_hl * 1.250  # H3
    r2 = pri_close + range_hl * 1.166  # H2
    r1 = pri_close + range_hl * 1.083  # H1
    
    # Support levels
    s1 = pri_close - range_hl * 1.083  # L1
    s2 = pri_close - range_hl * 1.166  # L2
    s3 = pri_close - range_hl * 1.250  # L3
    s4 = pri_close - range_hl * 1.500  # L4
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(pivot[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or 
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
            # Long conditions: price breaks above Camarilla H4 AND 1d HMA rising AND volume spike
            if (close[i] > r4[i] and 
                hma_rising_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L4 AND 1d HMA falling AND volume spike
            elif (close[i] < s4[i] and 
                  hma_falling_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to pivot (midline) OR opposite breakout occurs
            exit_long = (position == 1 and 
                        (close[i] < pivot[i] or close[i] < s4[i]))
            exit_short = (position == -1 and 
                         (close[i] > pivot[i] or close[i] > r4[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals