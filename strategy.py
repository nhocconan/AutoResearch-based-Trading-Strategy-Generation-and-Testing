#!/usr/bin/env python3
"""
Experiment #1881: 4h Donchian(20) Breakout + HMA Trend + Volume Confirmation + ATR Stoploss
HYPOTHESIS: Donchian channel breakouts capture strong momentum moves. Combined with 1d HMA(50) trend filter, volume confirmation (>1.5x average), and ATR-based stoploss, this strategy works in both bull and bear markets by following the higher timeframe trend. Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing of 0.25 to manage drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1881_4h_donchian20_hma_vol_v1"
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d HMA(50) for trend direction
    def calculate_hma(arr, period):
        half = arr.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
        sqrt = arr.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
        return 2*half - sqrt
    
    hma_50_1d = calculate_hma(pd.Series(close_1d), 50).values
    trend_1d = np.where(close_1d > hma_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian Channel(20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: HMA(21) for trend confirmation ===
    def calculate_hma_series(arr, period):
        half = arr.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
        sqrt = arr.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
        return (2*half - sqrt).values
    
    hma_21 = calculate_hma_series(pd.Series(close), 21)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    warmup = 50  # sufficient for Donchian(20) and HMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(hma_21[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss or reverse signal ---
        if in_position:
            # Stoploss: 2 * ATR against position
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * entry_atr:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian breakout in opposite direction
                elif price < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price > entry_price + 2.0 * entry_atr:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian breakout in opposite direction
                elif price > donchian_high[i]:
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
            # Long: Price breaks above Donchian high + price above HMA + 1d trend up
            if (price > donchian_high[i] and 
                price > hma_21[i] and 
                trend_bias > 0):
                in_position = True
                position_side = 1
                entry_price = price
                entry_atr = atr[i]
                signals[i] = SIZE
            # Short: Price breaks below Donchian low + price below HMA + 1d trend down
            elif (price < donchian_low[i] and 
                  price < hma_21[i] and 
                  trend_bias < 0):
                in_position = True
                position_side = -1
                entry_price = price
                entry_atr = atr[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals