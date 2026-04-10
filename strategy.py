#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 12h HMA trend filter + volume confirmation
# - Long when price breaks above Camarilla H3 level AND 12h HMA(21) is rising AND volume > 1.5x 20-period average
# - Short when price breaks below Camarilla L3 level AND 12h HMA(21) is falling AND volume > 1.5x 20-period average
# - Exit when price crosses Camarilla Pivot point (midline) OR opposite breakout occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Camarilla pivots provide strong intraday support/resistance levels
# - 12h HMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false breakouts

name = "4h_12h_camarilla_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Pre-compute 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 12h HMA(21) for trend filter
    close_12h = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, n):
        if len(arr) < n:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, n + 1)
        return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
    
    # Calculate WMA for n/2 and n
    n_hma = 21
    half_n = n_hma // 2
    sqrt_n = int(np.sqrt(n_hma))
    
    wma_half = wma(close_12h, half_n)
    wma_full = wma(close_12h, n_hma)
    
    # Handle array lengths
    wma_half_padded = np.full_like(close_12h, np.nan)
    wma_full_padded = np.full_like(close_12h, np.nan)
    
    if len(wma_half) > 0:
        wma_half_padded[half_n-1:half_n-1+len(wma_half)] = wma_half
    if len(wma_full) > 0:
        wma_full_padded[n_hma-1:n_hma-1+len(wma_full)] = wma_full
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half_padded - wma_full_padded
    # WMA(sqrt(n)) of the diff
    wma_diff = wma(diff, sqrt_n)
    wma_diff_padded = np.full_like(close_12h, np.nan)
    if len(wma_diff) > 0:
        wma_diff_padded[sqrt_n-1:sqrt_n-1+len(wma_diff)] = wma_diff
    
    hma_12h = wma_diff_padded
    
    # HMA slope (rising/falling)
    hma_slope = np.diff(hma_12h, prepend=np.nan)
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    # Align HTF indicators to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_12h, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_12h, hma_falling)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Calculate Camarilla pivots for current 4h bar using prior bar's OHLC
        if i == 0:
            # Not enough data for first bar
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
            
        # Use previous bar's OHLC to calculate today's Camarilla levels
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        # Camarilla levels
        h3 = pivot + (range_val * 1.1 / 4)
        l3 = pivot - (range_val * 1.1 / 4)
        h4 = pivot + (range_val * 1.1 / 2)
        l4 = pivot - (range_val * 1.1 / 2)
        
        # Skip if any required data is invalid
        if (np.isnan(pivot) or np.isnan(h3) or np.isnan(l3) or 
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
            # Long conditions: price breaks above Camarilla H3 level AND 12h HMA rising AND volume spike
            if (close[i] > h3 and 
                hma_rising_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 level AND 12h HMA falling AND volume spike
            elif (close[i] < l3 and 
                  hma_falling_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla Pivot point OR opposite breakout occurs
            exit_long = (position == 1 and 
                        (close[i] < pivot or close[i] < l3))
            exit_short = (position == -1 and 
                         (close[i] > pivot or close[i] > h3))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals