#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeRegime
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation (>1.5x 20-period median), and choppiness regime filter (CHOP > 61.8 = range, avoid breakouts in chop). 
Enters long when price breaks above 20-period high with volume confirmation, bullish 1d trend, and non-choppy market (CHOP <= 61.8). 
Enters short when price breaks below 20-period low with volume confirmation, bearish 1d trend, and non-choppy market. 
Uses ATR-based trailing stop (exit when price retracs 2.5x ATR from extreme) and discrete position sizing (0.25). 
Target: 20-50 trades/year. Works in both bull and bear markets by following 1d trend and avoiding false breakouts in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: volume > 1.5x 20-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    # ATR (14) for stoploss and regime
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14) for regime filter
    def choppiness_index(high, low, close, window=14):
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum()
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
        return chop.values
    chop = choppiness_index(high, low, close, 14)
    
    # Load 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Track extreme for trailing stop
    long_extreme = 0.0
    short_extreme = 0.0
    
    # Start after warmup (need 20-period Donchian, 20-period volume median, 14-period ATR/CHOP, 50-period EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_median[i]) or np.isnan(atr[i]) or 
            np.isnan(chop[i]) or np.isnan(ema50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above Donchian high + volume confirm + bullish 1d trend + non-choppy (CHOP <= 61.8)
        if (close[i] > donchian_high[i] and volume_confirm[i] and 
            close[i] > ema50_1d_aligned[i] and chop[i] <= 61.8):
            if position != 1:
                signals[i] = base_size
                position = 1
                long_extreme = close[i]  # reset extreme on new entry
            else:
                signals[i] = base_size
                # update long extreme
                if close[i] > long_extreme:
                    long_extreme = close[i]
        # Short logic: break below Donchian low + volume confirm + bearish 1d trend + non-choppy
        elif (close[i] < donchian_low[i] and volume_confirm[i] and 
              close[i] < ema50_1d_aligned[i] and chop[i] <= 61.8):
            if position != -1:
                signals[i] = -base_size
                position = -1
                short_extreme = close[i]  # reset extreme on new entry
            else:
                signals[i] = -base_size
                # update short extreme
                if close[i] < short_extreme:
                    short_extreme = close[i]
        # Exit conditions
        else:
            # Long exit: reversal signal OR trailing stop hit
            if position == 1:
                # Trailing stop: exit if price retracs 2.5* ATR from long extreme
                if close[i] < long_extreme - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Reversal: close below Donchian low OR bearish 1d trend OR choppy market
                elif (close[i] < donchian_low[i] or 
                      close[i] < ema50_1d_aligned[i] or 
                      chop[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = base_size
                    # update extreme
                    if close[i] > long_extreme:
                        long_extreme = close[i]
            # Short exit: reversal signal OR trailing stop hit
            elif position == -1:
                # Trailing stop: exit if price retracs 2.5* ATR from short extreme
                if close[i] > short_extreme + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Reversal: close above Donchian high OR bullish 1d trend OR choppy market
                elif (close[i] > donchian_high[i] or 
                      close[i] > ema50_1d_aligned[i] or 
                      chop[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -base_size
                    # update extreme
                    if close[i] < short_extreme:
                        short_extreme = close[i]
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeRegime"
timeframe = "4h"
leverage = 1.0