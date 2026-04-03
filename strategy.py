#!/usr/bin/env python3
"""
Experiment #123: 4h Donchian Breakout + 12h HMA Trend + Volume Spike

HYPOTHESIS: A 4h Donchian(20) breakout in the direction of the 12h HMA(21) trend, 
confirmed by a 12h volume spike (>2x average), captures high-probability trend 
continuation moves. The Donchian structure provides objective breakout levels, 
while the 12h HMA filter ensures alignment with the higher timeframe trend to 
avoid counter-trend whipsaws. Volume confirmation filters for institutional 
participation. Targets 19-50 trades/year (75-200 total over 4 years) on 4h timeframe 
to minimize fee drag while maintaining statistical validity. Designed to work in 
both bull and bear markets by only trading breakouts aligned with the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend and volume spike (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close
    if len(df_12h) >= 21:
        close_12h = df_12h['close'].values
        # HMA: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        wma_half = np.array([wma(close_12h[i:i+half_len], half_len) 
                            for i in range(len(close_12h) - half_len + 1)])
        wma_full = np.array([wma(close_12h[i:i+21], 21) 
                            for i in range(len(close_12h) - 21 + 1)])
        hma_2xhalf = 2 * wma_half
        hma_diff = hma_2xhalf[:len(wma_full)] - wma_full
        hma_21 = np.array([wma(hma_diff[i:i+sqrt_len], sqrt_len) 
                          for i in range(len(hma_diff) - sqrt_len + 1)])
        
        # Pad to match original length
        hma_21_padded = np.full(len(close_12h), np.nan)
        hma_21_padded[half_len-1:len(hma_21)+half_len-1] = hma_21
        hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21_padded)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === 4h Indicators ===
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, lookback * 2)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian lower band touch (mean reversion signal)
                if close[i] <= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian upper band touch (mean reversion signal)
                if close[i] >= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Determine 12h trend direction
        trend_up = hma_21_aligned[i] > close[i]  # Price above HMA = uptrend
        trend_down = hma_21_aligned[i] < close[i]  # Price below HMA = downtrend
        
        # Volume confirmation: require significant spike
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
        # Long: Donchian breakout above upper band in uptrend with volume
        long_condition = (
            close[i] > highest_high[i] and  # Breakout above Donchian high
            trend_up and                    # Aligned with 12h uptrend
            volume_spike                    # Volume confirmation
        )
        
        # Short: Donchian breakdown below lower band in downtrend with volume
        short_condition = (
            close[i] < lowest_low[i] and    # Breakdown below Donchian low
            trend_down and                  # Aligned with 12h downtrend
            volume_spike                    # Volume confirmation
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals