#!/usr/bin/env python3
"""
Experiment #252: 12h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts on 12h capture medium-term momentum. 1d HMA(21) defines trend regime (bullish when price > HMA). Volume confirmation (>1.5x average) filters false breakouts. ATR(14) stoploss (2.5x) manages risk. Discrete position sizing (0.25) controls fee drag. Designed for 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year). Works in bull markets via trend-following breakouts and in bear markets via mean-reversion fading of failed breakouts (when price < HMA and breaks lower band).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_252_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA(21) trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        wma_2xhalf = 2 * wma_half
        diff = wma_2xhalf - wma_full
        return wma(diff, sqrt)
    hma_21 = hma(close_1d, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    # === 12h Indicators: Donchian(20) channels ===
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    
    # === 12h Indicators: ATR(14) for stoploss and volume filter ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period Donchian and 21-period HMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Trend Regime: 1d HMA(21) ---
        bullish_regime = price > hma_21_aligned[i]
        bearish_regime = price < hma_21_aligned[i]
        
        # --- Donchian Breakout Signals ---
        upper_breakout = price > donch_high[i-1]  # break above previous high
        lower_breakout = price < donch_low[i-1]   # break below previous low
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit conditions: failed breakout or reversal
                if bearish_regime and lower_breakout and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit conditions: failed breakout or reversal
                if bullish_regime and upper_breakout and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if volume_spike:
            # Bull regime: go long on upper breakout
            if bullish_regime and upper_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Bear regime: go short on lower breakout
            elif bearish_regime and lower_breakout:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            # Mean reversion in opposing regime: fade failed breakouts
            elif bullish_regime and lower_breakout:
                # Failed lower breakout in bull regime -> long reversion
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif bearish_regime and upper_breakout:
                # Failed upper breakout in bear regime -> short reversion
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals