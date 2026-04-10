#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with weekly trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 (1d) AND weekly HMA(34) is rising AND volume > 1.5x 20-period average
# - Short when price breaks below Camarilla L3 (1d) AND weekly HMA(34) is falling AND volume > 1.5x 20-period average
# - Exit when price crosses Camarilla Pivot point (1d) OR opposite breakout occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots provide institutional support/resistance levels
# - Weekly HMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false breakouts

name = "12h_1w_camarilla_hma_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 5 or len(df_1w) < 34:
        return np.zeros(n)
    
    # Pre-compute 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d Camarilla levels (based on previous day)
    # Camarilla: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # Using previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Pre-compute 1w HMA(34) for trend filter
    close_1w = df_1w['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, n):
        if len(arr) < n:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, n + 1)
        return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
    
    # Calculate WMA for n/2 and n
    n = 34
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
    
    # Align HTF indicators to 12h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND weekly HMA rising AND volume spike
            if (close[i] > camarilla_h3_aligned[i] and 
                hma_rising_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND weekly HMA falling AND volume spike
            elif (close[i] < camarilla_l3_aligned[i] and 
                  hma_falling_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla Pivot OR opposite breakout occurs
            exit_long = (position == 1 and 
                        (close[i] < camarilla_pivot_aligned[i] or close[i] < camarilla_l3_aligned[i]))
            exit_short = (position == -1 and 
                         (close[i] > camarilla_pivot_aligned[i] or close[i] > camarilla_h3_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals