#!/usr/bin/env python3
"""
Experiment #053: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Confirmation + ATR Stoploss

HYPOTHESIS: Donchian(20) breakouts on 4h capture medium-term trends when aligned with 
12h HMA(21) trend direction and confirmed by 4h volume spikes (>1.8x 20-period average). 
ATR-based stoploss (2.5x ATR) limits downside. This structure works in both bull 
and bear markets by requiring trend alignment (HMA) and volume confirmation to filter 
false breakouts. Target: 75-200 trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 21:
        close_12h = df_12h['close'].values
        # HMA(21): WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half = 21 // 2
        sqrt_n = int(np.sqrt(21))
        wma_half = pd.Series(close_12h).ewm(span=half, adjust=False).mean().values
        wma_full = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_12h = pd.Series(raw_hma).ewm(span=sqrt_n, adjust=False).mean().values
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Donchian(20) channels ===
    donchian_len = 20
    if n >= donchian_len:
        upper = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
        lower = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    else:
        upper = lower = np.full(n, np.nan)
    
    # === 4h volume confirmation (>1.8x 20-period average) ===
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio = np.zeros(n)
        vol_ratio[20:] = volume[20:] / vol_ma[20:]
        vol_ratio[:20] = 1.0  # Neutral for warmup
    else:
        vol_ratio = np.full(n, 1.0)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ratio[i])):
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trend filter: price > 12h HMA = uptrend, price < 12h HMA = downtrend
        uptrend = close[i] > hma_12h_aligned[i]
        downtrend = close[i] < hma_12h_aligned[i]
        
        # Volume confirmation: 4h volume spike > 1.8x average
        volume_spike = vol_ratio[i] > 1.8
        
        # Breakout logic: Donchian breakout in trend direction with volume
        if uptrend and volume_spike:
            # Long breakout above upper Donchian
            if close[i] > upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
        
        elif downtrend and volume_spike:
            # Short breakdown below lower Donchian
            if close[i] < lower[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
        
        # No signal
        else:
            signals[i] = 0.0
    
    return signals