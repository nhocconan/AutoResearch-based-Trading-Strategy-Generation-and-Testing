#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + HMA(21) trend filter + volume confirmation
# Long when price breaks above Donchian upper (20) AND HMA(21) upward sloping AND volume spike
# Short when price breaks below Donchian lower (20) AND HMA(21) downward sloping AND volume spike
# Uses Donchian channels for structure, HMA for smooth trend filter, volume for conviction.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation).
# Timeframe: 4h (proven timeframe for balance of signal quality and trade frequency).

name = "4h_Donchian20_HMA21_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for additional trend filter (optional)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for additional trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    if len(high) >= 20:
        # Upper channel: highest high over past 20 periods
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Lower channel: lowest low over past 20 periods
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
    
    # Calculate HMA(21) on 4h close
    if len(close) >= 21:
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        # WMA function for HMA calculation
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        # Calculate HMA: WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
        wma_half = np.array([np.nan] * half_len + [wma(close[i:i+half_len], half_len) 
                           for i in range(len(close) - half_len + 1)])
        wma_full = np.array([np.nan] * 21 + [wma(close[i:i+21], 21) 
                           for i in range(len(close) - 21 + 1)])
        
        # Handle edge cases for convolution
        if len(close) >= half_len:
            wma_half_vals = np.array([wma(close[i:i+half_len], half_len) 
                                     for i in range(len(close) - half_len + 1)])
            wma_half[half_len-1:len(wma_half_vals)+half_len-1] = wma_half_vals
        
        if len(close) >= 21:
            wma_full_vals = np.array([wma(close[i:i+21], 21) 
                                     for i in range(len(close) - 21 + 1)])
            wma_full[20:len(wma_full_vals)+20] = wma_full_vals
        
        # Calculate raw HMA: 2*WMA(half) - WMA(full)
        raw_hma = 2 * wma_half - wma_full
        
        # Final HMA: WMA(raw_hma, sqrt_len)
        if len(raw_hma) >= sqrt_len:
            hma_values = np.array([np.nan] * (sqrt_len - 1) + 
                                 [wma(raw_hma[i:i+sqrt_len], sqrt_len) 
                                  for i in range(len(raw_hma) - sqrt_len + 1)])
            hma_values[sqrt_len-1:len(hma_values)] = hma_values[sqrt_len-1:]  # This line is wrong, fix below
            
            # Correct calculation
            hma_raw = 2 * wma_half - wma_full
            hma_final = np.full(n, np.nan)
            valid_start = max(half_len, 21) - 1
            for i in range(valid_start, len(hma_raw) - sqrt_len + 1):
                if not (np.isnan(hma_raw[i:i+sqrt_len]).any()):
                    window_data = hma_raw[i:i+sqrt_len]
                    weights = np.arange(1, sqrt_len + 1)
                    hma_final[i + sqrt_len - 1] = np.dot(window_data, weights) / weights.sum()
        else:
            hma_final = np.full(n, np.nan)
    else:
        hma_final = np.full(n, np.nan)
    
    # Simpler HMA calculation using pandas (more reliable)
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    close_series = pd.Series(close)
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    
    def wma_pandas(series, window):
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, np.arange(1, window+1)) / np.arange(1, window+1).sum(), raw=True
        )
    
    wma_half = wma_pandas(close_series, half_n)
    wma_full = wma_pandas(close_series, 21)
    hma_raw = 2 * wma_half - wma_full
    hma_21 = wma_pandas(hma_raw, sqrt_n).values
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Higher threshold for fewer trades
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_21[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND HMA rising AND volume spike
            # AND close > 1d EMA34 (additional uptrend filter)
            if (close[i] > highest_high[i] and 
                hma_21[i] > hma_21[i-1] and  # HMA rising
                volume_spike[i] and
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND HMA falling AND volume spike
            # AND close < 1d EMA34 (additional downtrend filter)
            elif (close[i] < lowest_low[i] and 
                  hma_21[i] < hma_21[i-1] and  # HMA falling
                  volume_spike[i] and
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian upper OR HMA turns down
            if close[i] < highest_high[i] or hma_21[i] < hma_21[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian lower OR HMA turns up
            if close[i] > lowest_low[i] or hma_21[i] > hma_21[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals