#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h HMA trend filter and volume confirmation
# - Long when price breaks above H3 level AND 12h HMA21 rising AND volume > 2.0x 20-bar avg
# - Short when price breaks below L3 level AND 12h HMA21 falling AND volume > 2.0x 20-bar avg
# - Exit when price returns to Pivot level (mean reversion to equilibrium)
# - Uses 12h HMA21 for trend filter (less lag than EMA, better for trend detection)
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 25-40 trades/year on 4h timeframe (100-160 total over 4 years)
# - Camarilla pivots work well in ranging markets; HMA trend filter adds directional bias in trends

name = "4h_12h_camarilla_breakout_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from daily data
    # Using 1d data for pivot calculation (standard practice)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # H3, L3 levels (most important for breakouts)
    h3 = close_1d + (range_1d * 1.1 / 4)
    l3 = close_1d - (range_1d * 1.1 / 4)
    # Pivot level for exit
    piv = pivot
    
    # Align HTF levels to LTF
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    piv_aligned = align_htf_to_ltf(prices, df_1d, piv)
    
    # Pre-compute 12h HMA(21) for trend filter (Hull Moving Average)
    close_12h = df_12h['close'].values
    # HMA formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    if len(close_12h) >= 21:
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, 21)
        # Handle array lengths
        wma_2x_half = 2 * wma_half
        # Pad to same length as wma_full
        if len(wma_2x_half) > len(wma_full):
            wma_2x_half = wma_2x_half[:len(wma_full)]
        elif len(wma_2x_half) < len(wma_full):
            wma_full = wma_full[:len(wma_2x_half)]
        diff = wma_2x_half - wma_full
        if len(diff) >= sqrt_len:
            hma_12h = wma(diff, sqrt_len)
        else:
            hma_12h = np.full_like(close_12h, np.nan)
        # Pad beginning with NaN
        hma_12h = np.concatenate([np.full(len(close_12h) - len(hma_12h), np.nan), hma_12h])
    else:
        hma_12h = np.full_like(close_12h, np.nan)
    
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(piv_aligned[i]) or np.isnan(hma_12h_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 12h HMA rising with volume spike
            if (prices['close'].iloc[i] > h3_aligned[i] and 
                hma_12h_aligned[i] > hma_12h_aligned[max(i-1, 0)] and  # HMA rising
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND 12h HMA falling with volume spike
            elif (prices['close'].iloc[i] < l3_aligned[i] and 
                  hma_12h_aligned[i] < hma_12h_aligned[max(i-1, 0)] and  # HMA falling
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot (mean reversion)
            # Exit when price returns to pivot level
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= piv_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= piv_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals