#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation
# - Long when price breaks above Donchian(20) high AND 1d HMA(21) rising AND 4h volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low AND 1d HMA(21) falling AND 4h volume > 1.5x 20-bar avg
# - Exit when price crosses Donchian(20) midline (mean of 20-period high/low)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian captures structural breaks; 1d HMA filter ensures alignment with daily trend
# - Volume confirmation avoids low-liquidity false breakouts
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Works in both bull and bear markets: trend filter prevents counter-trend trades in bear, breakouts capture momentum in bull

name = "4h_1d_donchian_breakout_hma_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d HMA(21) trend: rising/falling
    close_1d = df_1d['close'].values
    # HMA formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(data, window):
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights, 'valid') / weights.sum()
    
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    wma_half = np.array([wma(close_1d[i:i+half_len], half_len) if i+half_len <= len(close_1d) else np.nan 
                         for i in range(len(close_1d))])
    wma_full = np.array([wma(close_1d[i:i+21], 21) if i+21 <= len(close_1d) else np.nan 
                         for i in range(len(close_1d))])
    wma_2x_half = 2 * wma_half
    diff = wma_2x_half - wma_full
    
    hma = np.array([wma(diff[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(diff) else np.nan 
                    for i in range(len(diff))])
    
    # HMA trend: rising if current > previous, falling if current < previous
    hma_rising = np.zeros_like(hma, dtype=bool)
    hma_falling = np.zeros_like(hma, dtype=bool)
    hma_rising[1:] = hma[1:] > hma[:-1]
    hma_falling[1:] = hma[1:] < hma[:-1]
    
    # Align 1d HMA trend to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling)
    
    # Pre-compute Donchian(20) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Donchian breakout conditions
    breakout_up = close > highest_high
    breakout_down = close < lowest_low
    
    # Exit when price crosses Donchian midline
    exit_long = close < donchian_mid
    exit_short = close > donchian_mid
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or
            np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(exit_long[i]) or np.isnan(exit_short[i]) or
            np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when Donchian breakout up AND 1d HMA rising AND volume spike
            if (breakout_up[i] and 
                hma_rising_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when Donchian breakout down AND 1d HMA falling AND volume spike
            elif (breakout_down[i] and 
                  hma_falling_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at Donchian midline
            # Exit when price crosses Donchian midline
            if position == 1 and exit_long[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and exit_short[i]:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals