#!/usr/bin/env python3
"""
Experiment #928: 12h Donchian(20) + 1w HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: Donchian breakouts on 12h capture medium-term momentum, filtered by 1w HMA trend direction 
and volume confirmation (>1.8x average). Long when price breaks above Donchian upper 
AND 1w HMA rising AND volume spike. Short when price breaks below Donchian lower 
AND 1w HMA falling AND volume spike. Uses discrete position sizing (0.25) to balance 
risk and reward. Target: 75-150 total trades over 4 years (19-37/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_928_12h_donchian20_1w_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate HMA(21) on 1w
    def calculate_hma(arr, period):
        half = int(period / 2)
        sqrt = int(np.sqrt(period))
        wma1 = pd.Series(arr).ewm(span=half, min_periods=half, adjust=False).mean().values
        wma2 = pd.Series(arr).ewm(span=period, min_periods=period, adjust=False).mean().values
        raw_hma = 2 * wma1 - wma2
        hma = pd.Series(raw_hma).ewm(span=sqrt, min_periods=sqrt, adjust=False).mean().values
        return hma
    
    hma_1w = calculate_hma(close_1w, 21)
    # Trend: 1 = rising (hma > previous hma), -1 = falling (hma < previous hma), 0 = flat
    hma_trend_1w = np.zeros_like(hma_1w)
    hma_trend_1w[1:] = np.where(hma_1w[1:] > hma_1w[:-1], 1, 
                                 np.where(hma_1w[1:] < hma_1w[:-1], -1, 0))
    # Align trend to 12h timeframe
    hma_trend_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_trend_1w)
    
    # === 12h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
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
    
    warmup = max(20, 20)  # sufficient for Donchian, volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(hma_trend_1w_aligned[i]) or
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
            
            # Optional: time-based exit after 8 bars (~4d on 12h) to avoid overtrading
            if bars_since_entry > 8:
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
        
        if volume_spike:
            # Long: price breaks above Donchian upper AND 1w HMA rising
            if price > upper_20[i] and hma_trend_1w_aligned[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower AND 1w HMA falling
            elif price < lower_20[i] and hma_trend_1w_aligned[i] < 0:
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