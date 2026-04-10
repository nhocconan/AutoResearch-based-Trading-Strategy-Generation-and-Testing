#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND 1d HMA(21) rising (uptrend) AND 4h volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low AND 1d HMA(21) falling (downtrend) AND 4h volume > 1.5x 20-bar avg
# - Exit when price crosses opposite Donchian level or HMA trend reverses
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian captures breakouts; 1d HMA filter ensures alignment with daily trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in both bull and bear markets: breakouts occur in all regimes, trend filter prevents counter-trend trades

name = "4h_1d_donchian_hma_volume_trend_v1"
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
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    # Calculate HMA: WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = np.array([wma(close_1d[i-half_length+1:i+1], half_length) if i >= half_length-1 else np.nan 
                         for i in range(len(close_1d))])
    wma_full = np.array([wma(close_1d[i-21+1:i+1], 21) if i >= 20 else np.nan 
                         for i in range(len(close_1d))])
    raw_hma = 2 * wma_half - wma_full
    hma = np.array([wma(raw_hma[i-sqrt_length+1:i+1], sqrt_length) if i >= sqrt_length-1 else np.nan 
                    for i in range(len(raw_hma))])
    
    # HMA trend: rising if current > previous, falling if current < previous
    hma_rising = np.zeros_like(hma, dtype=bool)
    hma_falling = np.zeros_like(hma, dtype=bool)
    for i in range(1, len(hma)):
        if not np.isnan(hma[i]) and not np.isnan(hma[i-1]):
            hma_rising[i] = hma[i] > hma[i-1]
            hma_falling[i] = hma[i] < hma[i-1]
    
    # Align 1d HMA trend to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling)
    
    # Pre-compute Donchian(20) channels on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
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
            # Long when price breaks above Donchian high AND 1d HMA rising AND volume spike
            if (close[i] > highest_high[i] and 
                hma_rising_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND 1d HMA falling AND volume spike
            elif (close[i] < lowest_low[i] and 
                  hma_falling_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit when price crosses opposite Donchian level or HMA trend reverses
            exit_signal = False
            if position == 1:  # Long position
                if close[i] < lowest_low[i] or not hma_rising_aligned[i]:
                    exit_signal = True
            else:  # Short position
                if close[i] > highest_high[i] or not hma_falling_aligned[i]:
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