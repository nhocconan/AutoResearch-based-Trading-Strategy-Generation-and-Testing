#!/usr/bin/env python3
"""
Experiment #1446: 4h Donchian(20) Breakout + 1d Trend + Volume Confirmation + Chop Filter
HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture intermediate-term trends with controlled trade frequency (target: 75-200 total over 4 years). 
Trend filter from 1d timeframe ensures alignment with higher-timeframe momentum. Volume confirmation (>1.8x average) filters for institutional participation. 
Choppiness Index regime filter avoids whipsaws in ranging markets (CHOP > 61.8 = range, avoid breakouts; CHOP < 38.2 = trend, allow breakouts). 
Designed to work in both bull (breakouts continue) and bear (breakdowns continue) markets by following the 1d trend direction. 
Uses ATR-based stoploss for risk management. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1446_4h_donchian20_1d_trend_vol_chop_v1"
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
    # Simple trend: price > previous close = uptrend, < = downtrend
    trend_1d = np.zeros(len(close_1d))
    trend_1d[1:] = np.where(close_1d[1:] > close_1d[:-1], 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    # === 4h Indicators: Choppiness Index (CHOP) for regime filter ===
    # CHOP = 100 * log10(sum(ATR(1)) / (HHV(high,14) - LLV(low,14))) / log10(14)
    atr1 = np.zeros(n)
    for i in range(1, n):
        atr1[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr1[0] = high[0] - low[0]
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.full(n, 50.0)  # default neutral
    denominator = hh - ll
    # Avoid division by zero
    mask = (denominator > 0) & (~np.isnan(denominator)) & (~np.isnan(sum_atr1))
    chop[mask] = 100 * np.log10(sum_atr1[mask] / denominator[mask]) / np.log10(14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for Donchian, volume MA, ATR, CHOP
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(chop[i])):
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
        # Regime filter: only allow breakouts in trending markets (CHOP < 38.2)
        trending_market = chop[i] < 38.2
        
        if volume_spike and trending_market:
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