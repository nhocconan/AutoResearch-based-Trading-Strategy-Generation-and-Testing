#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d HMA(21) trend + Volume confirmation
# Long when price breaks above 20-bar Donchian high AND price > 1d HMA21 AND volume > 1.8x 20-bar avg
# Short when price breaks below 20-bar Donchian low AND price < 1d HMA21 AND volume > 1.8x 20-bar avg
# Exit via ATR-based trailing stop: signal=0 when long and price < highest_high - 2.5*ATR(14) or short and price > lowest_low + 2.5*ATR(14)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-30 trades/year on 12h timeframe.
# Donchian provides structure, 1d HMA21 filters counter-trend moves, volume confirmation ensures breakout strength.
# This combination has worked well across multiple timeframes and assets in bear/bull markets.

name = "12h_Donchian20_1dHMA21_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate HMA(21) on 1d data: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    close_1d = df_1d['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    wma_2x_sub = 2 * wma_half - wma_full
    hma_21_1d = wma(wma_2x_sub, sqrt_len)
    
    # Pad HMA array to match df_1d length
    hma_21_1d_padded = np.full(len(close_1d), np.nan)
    hma_21_1d_padded[half_len:half_len + len(hma_21_1d)] = hma_21_1d
    
    # Align HMA21 to 12h timeframe
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d_padded)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34)  # Donchian, volume MA, ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(hma_21_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr_14[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_hma = hma_21_1d_aligned[i]
        curr_highest = highest_high[i]
        curr_lowest = lowest_low[i]
        curr_atr = atr_14[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # ATR trailing stop: exit when price < highest_high - 2.5*ATR
            if curr_close < curr_highest - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # ATR trailing stop: exit when price > lowest_low + 2.5*ATR
            if curr_close > curr_lowest + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND price > 1d HMA21 AND volume confirmation
            if curr_close > curr_highest and curr_close > curr_hma and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short when price breaks below Donchian low AND price < 1d HMA21 AND volume confirmation
            elif curr_close < curr_lowest and curr_close < curr_hma and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals