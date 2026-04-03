#!/usr/bin/env python3
"""
Experiment #1422: 12h Donchian(20) Breakout + 1d Trend + Volume Confirmation + Chop Filter
HYPOTHESIS: Donchian(20) breakouts on 12h timeframe capture intermediate-term trends with low trade frequency (target: 50-150 total over 4 years). 
Trend filter from 1d timeframe ensures alignment with higher-timeframe momentum. Volume confirmation (>1.8x average) filters for institutional participation. 
Choppiness regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trending) avoids whipsaws in sideways markets. 
Designed to work in both bull (breakouts continue) and bear (breakdowns continue) markets by following the 1d trend direction. 
Uses ATR-based stoploss for risk management. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1422_12h_donchian20_1d_trend_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter and chop regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Trend: price > previous close = uptrend, < = downtrend
    trend_1d = np.zeros(len(close_1d))
    trend_1d[1:] = np.where(close_1d[1:] > close_1d[:-1], 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 1d ATR(14) for Chop calculation
    tr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # 1d Chopiness Index (CHOP) = 100 * log10(sum(ATR(14)) / (max(high)-min(low))) / log10(14)
    chop_1d = np.full(len(close_1d), 50.0)  # default neutral
    for i in range(14, len(close_1d)):
        sum_atr = np.sum(atr_1d[i-13:i+1])
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        if max_high > min_low:
            chop_1d[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
        else:
            chop_1d[i] = 50.0
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 12h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        # Chop filter: only trade in trending regimes (CHOP < 38.2) OR strong breakouts in chop
        chop_trending = chop_1d_aligned[i] < 38.2
        
        if volume_spike:
            # Breakout: price breaks above upper band OR below lower band
            if price > donch_high[i] and trend_1d_aligned[i] > 0:  # 1d uptrend
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donch_low[i] and trend_1d_aligned[i] < 0:  # 1d downtrend
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