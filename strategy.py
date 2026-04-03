#!/usr/bin/env python3
"""
Experiment #1881: 4h Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian(20) breakouts capture strong momentum moves. Combined with HMA(21) trend filter (4h) and 1d EMA(50) for higher timeframe alignment, volume confirmation (>1.5x average), and ATR-based stoploss (2*ATR), this strategy avoids false breakouts and chops. Works in both bull and bear markets by following the 1d trend direction. Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing of 0.30.
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
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: HMA(21) for trend ===
    def hma(series, period):
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma = lambda s, p: pd.Series(s).ewm(span=p, adjust=False).mean().values
        wma_half = wma(series, half)
        wma_full = wma(series, period)
        raw = 2 * wma_half - wma_full
        return wma(raw, sqrt)
    
    hma_21 = hma(close, 21)
    hma_trend = np.where(close > hma_21, 1, -1)
    
    # === 4h Indicators: Donchian(20) channels ===
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20), EMA(50), ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21[i]) or np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR stoploss or trend reversal ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2*ATR below entry
                if price < entry_price - 2.0 * atr[i]:
                    exit_signal = True
                # Exit if 4h trend flips
                elif hma_trend[i] < 0:
                    exit_signal = True
                # Exit if 1d trend flips
                elif trend_1d_aligned[i] < 0:
                    exit_signal = True
            else:  # Short position
                # Stoploss: 2*ATR above entry
                if price > entry_price + 2.0 * atr[i]:
                    exit_signal = True
                # Exit if 4h trend flips
                elif hma_trend[i] > 0:
                    exit_signal = True
                # Exit if 1d trend flips
                elif trend_1d_aligned[i] > 0:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Require 4h HMA trend alignment
        hma_bias = hma_trend[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        # Only trade if 1d and 4h trends align
        if trend_bias == hma_bias and volume_spike:
            # Long: price breaks above Donchian upper channel
            if price > dc_upper[i] and trend_bias > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower channel
            elif price < dc_lower[i] and trend_bias < 0:
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