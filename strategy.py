#!/usr/bin/env python3
"""
Experiment #3830: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Daily Donchian breakouts capture multi-week swings with 1-week HMA (21) confirming trend alignment. Volume > 1.3x average confirms institutional participation. Works in bull markets (breakouts above resistance) and bear markets (breakdowns below support). Discrete position sizing (0.25) minimizes fee drag. Target: 50-120 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3830_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA(21) trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate HMA(21) on weekly close
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half_period, adjust=False).mean().values
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw_hma = 2 * wma2 - wma1
        hma_vals = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
        return hma_vals
    
    hma_21_1w = hma(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # === 1d Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                # Calculate ATR(14) on the fly for exit condition
                if i >= 14:
                    tr = np.maximum(high[i] - low[i], 
                                  np.maximum(np.abs(high[i] - close[i-1]), 
                                             np.abs(low[i] - close[i-1])))
                    atr_14 = np.mean([np.maximum(high[j] - low[j], 
                                               np.maximum(np.abs(high[j] - close[j-1]), 
                                                      np.abs(low[j] - close[j-1])) 
                                      for j in range(max(1, i-13), i+1)])
                else:
                    atr_14 = 0.0
                if price < highest_since_entry - 2.0 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if i >= 14:
                    tr = np.maximum(high[i] - low[i], 
                                  np.maximum(np.abs(high[i] - close[i-1]), 
                                             np.abs(low[i] - close[i-1])))
                    atr_14 = np.mean([np.maximum(high[j] - low[j], 
                                               np.maximum(np.abs(high[j] - close[j-1]), 
                                                      np.abs(low[j] - close[j-1])) 
                                      for j in range(max(1, i-13), i+1)])
                else:
                    atr_14 = 0.0
                if price > lowest_since_entry + 2.0 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.3x average)
        volume_spike = vol_ratio[i] > 1.3
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND above weekly HMA (bullish breakout with trend alignment)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                price > hma_21_1w_aligned[i]):  # Above weekly HMA (bullish trend)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below weekly HMA (bearish breakdown with trend alignment)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < hma_21_1w_aligned[i]):  # Below weekly HMA (bearish trend)
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals