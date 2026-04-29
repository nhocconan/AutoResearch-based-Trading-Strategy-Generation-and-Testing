#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend filter + volume confirmation
# Donchian breakouts capture strong momentum moves; 1d HMA ensures alignment with higher-timeframe trend;
# volume confirmation filters false breakouts. Works in bull markets via continuation breakouts
# and in bear markets via mean-reversion failures at bands. Target: 20-50 trades/year (80-200 total).

name = "4h_Donchian20_Breakout_1dHMA21_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d HMA(21) - Hull Moving Average
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    close_1d = df_1d['close'].values
    n_1d = len(close_1d)
    if n_1d < 21:
        hma_1d = np.full(n_1d, np.nan)
    else:
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        wma_half = np.array([wma(close_1d[i:i+half_n], half_n) if i+half_n <= n_1d else np.nan 
                            for i in range(n_1d - half_n + 1)])
        wma_full = np.array([wma(close_1d[i:i+21], 21) if i+21 <= n_1d else np.nan 
                            for i in range(n_1d - 21 + 1)])
        wma_2x_sub = 2 * wma_half - wma_full[:len(wma_half)]
        hma_1d = np.array([wma(wma_2x_sub[i:i+sqrt_n], sqrt_n) if i+sqrt_n <= len(wma_2x_sub) else np.nan 
                          for i in range(len(wma_2x_sub) - sqrt_n + 1)])
        # Pad to match original length
        hma_1d = np.concatenate([np.full(n_1d - len(hma_1d), np.nan), hma_1d])
    
    # Align 1d HMA to 4h timeframe (completed 1d bar only)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for 1d EMA, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(hma_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_hma_1d = hma_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price closes above upper Donchian + above 1d HMA + volume
            if curr_close > curr_upper and curr_close > curr_hma_1d and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below lower Donchian + below 1d HMA + volume
            elif curr_close < curr_lower and curr_close < curr_hma_1d and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price closes below lower Donchian OR below 1d HMA
            if curr_close < curr_lower or curr_close < curr_hma_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price closes above upper Donchian OR above 1d HMA
            if curr_close > curr_upper or curr_close > curr_hma_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals