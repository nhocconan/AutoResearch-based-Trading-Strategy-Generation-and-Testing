#!/usr/bin/env python3
"""
Experiment #1861: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture strong momentum moves. HMA(21) filters for trend direction, volume confirmation ensures institutional participation, and ATR stoploss manages risk. Works in both bull and bear markets by following the 1d trend. Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing of 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1861_4h_donchian20_hma21_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    # Upper channel: highest high of last 20 periods
    # Lower channel: lowest low of last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: HMA(21) for trend ===
    # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    def wma(series, period):
        weights = np.arange(1, period + 1)
        return np.convolve(series, weights/weights.sum(), mode='same')
    
    def hma(series, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = wma(series, half_period)
        wma_full = wma(series, period)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, sqrt_period)
    
    hma_21 = hma(close, 21)
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    warmup = 50  # sufficient for Donchian(20), HMA(21), ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_21[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR stoploss or reverse signal ---
        if in_position:
            # Stoploss conditions
            if position_side > 0:  # Long position
                if price < entry_price - 2.0 * entry_atr:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks below Donchian lower channel
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if HMA trend flips
                elif price < hma_21[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if price > entry_price + 2.0 * entry_atr:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks above Donchian upper channel
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if HMA trend flips
                elif price > hma_21[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper channel + HMA uptrend + 1d uptrend
            if (price > highest_high[i] and 
                hma_21[i] > hma_21[i-1] and  # HMA rising
                trend_bias > 0):
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower channel + HMA downtrend + 1d downtrend
            elif (price < lowest_low[i] and 
                  hma_21[i] < hma_21[i-1] and  # HMA falling
                  trend_bias < 0):
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals