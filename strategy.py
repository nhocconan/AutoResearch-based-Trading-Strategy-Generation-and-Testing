#!/usr/bin/env python3
"""
Experiment #873: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts on 4h capture medium-term momentum, filtered by 12h HMA trend direction
to align with higher timeframe momentum. Volume confirmation (>2.0x average) ensures breakout validity.
ATR-based stoploss (2.0x ATR) manages risk. Works in bull/bear markets: in bull trends, 12h HMA rising
filters for longs; in bear trends, 12h HMA falling filters for shorts. Discrete position sizing (0.25)
minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_873_4h_donchian20_12h_hma_vol_v1"
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
    close_12h = df_12h['close'].values
    
    # Calculate HMA(21) on 12h
    def calculate_hma(arr, period):
        half = int(period / 2)
        sqrt = int(np.sqrt(period))
        wma1 = pd.Series(arr).ewm(span=half, min_periods=half, adjust=False).mean().values
        wma2 = pd.Series(arr).ewm(span=period, min_periods=period, adjust=False).mean().values
        raw_hma = 2 * wma1 - wma2
        hma = pd.Series(raw_hma).ewm(span=sqrt, min_periods=sqrt, adjust=False).mean().values
        return hma
    
    hma_12h = calculate_hma(close_12h, 21)
    # Trend: 1 = rising (hma > previous hma), -1 = falling (hma < previous hma), 0 = flat
    hma_trend_12h = np.zeros_like(hma_12h)
    hma_trend_12h[1:] = np.where(hma_12h[1:] > hma_12h[:-1], 1, 
                                 np.where(hma_12h[1:] < hma_12h[:-1], -1, 0))
    # Align trend to 4h timeframe
    hma_trend_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_trend_12h)
    
    # === 4h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
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
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20)  # sufficient for Donchian, volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(hma_trend_12h_aligned[i]) or
            np.isnan(atr[i])):
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
            
            # Optional: time-based exit after 6 bars (~24h on 4h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long: price breaks above Donchian upper AND 12h HMA rising
            if price > upper_20[i] and hma_trend_12h_aligned[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower AND 12h HMA falling
            elif price < lower_20[i] and hma_trend_12h_aligned[i] < 0:
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