#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
# Long when price breaks above Donchian upper band(20) AND 12h HMA21 uptrend AND volume > 1.5x 20-period median.
# Short when price breaks below Donchian lower band(20) AND 12h HMA21 downtrend AND volume > 1.5x 20-period median.
# Donchian provides clear structure, 12h HMA filters higher-timeframe trend, volume confirms breakout strength.
# Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets.
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to minimize fee drag.

name = "4h_Donchian20_Breakout_12hHMA21_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 12h HMA(21) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    close_12h = df_12h['close'].values
    n_half = 21 // 2
    n_sqrt = int(np.sqrt(21))
    
    wma_full = wma(close_12h, 21)
    wma_half = wma(close_12h, n_half)
    wma_diff = 2 * wma_half - wma_full
    hma_21 = wma(wma_diff, n_sqrt)
    
    # Pad HMA array to match df_12h length
    hma_21_padded = np.full_like(close_12h, np.nan)
    hma_21_padded[n_half + n_sqrt - 1:] = hma_21
    
    # Align 12h HMA to 4h timeframe (wait for completed 12h bar)
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21_padded)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Donchian, HMA, and volume
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(hma_21_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 12h HMA21 direction
        uptrend = curr_close > hma_21_aligned[i]
        downtrend = curr_close < hma_21_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band AND uptrend AND volume spike
            if curr_close > highest_high[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower band AND downtrend AND volume spike
            elif curr_close < lowest_low[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price re-enters below Donchian upper band (mean reversion) OR trend turns down
            if curr_close < highest_high[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price re-enters above Donchian lower band (mean reversion) OR trend turns up
            if curr_close > lowest_low[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals