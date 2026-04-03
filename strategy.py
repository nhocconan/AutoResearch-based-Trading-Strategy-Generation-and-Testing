#!/usr/bin/env python3
"""
Experiment #713: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation
HYPOTHESIS: 4h Donchian breakout in direction of 12h HMA trend with volume spike captures 
institutional breakouts while avoiding false signals. Uses discrete position sizing (0.25) 
to minimize fee churn. Works in bull/bear markets via 12h HMA trend filter: long only when 
price above HMA, short only when price below HMA. Target: 75-200 total trades over 4 years 
(19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_713_4h_donchian20_12h_hma_vol_v1"
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
    
    # Calculate HMA(21) for 12h timeframe
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA calculation
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        wma_half = pd.Series(arr).rolling(window=half_period, min_periods=half_period).apply(
            lambda x: np.dot(x, weights[:half_period]) / weights[:half_period].sum(), raw=True
        ).values
        wma_full = pd.Series(arr).rolling(window=period, min_periods=period).apply(
            lambda x: np.dot(x, weights[:period]) / weights[:period].sum(), raw=True
        ).values
        
        raw_hma = 2 * wma_half - wma_full
        hma_values = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
            lambda x: np.dot(x, weights[:sqrt_period]) / weights[:sqrt_period].sum(), raw=True
        ).values
        
        return hma_values
    
    # Simplified HMA calculation using EMA approximation for performance
    # HMA ≈ EMA of (2*EMA(n/2) - EMA(n))
    ema_half = pd.Series(close_12h).ewm(span=10, min_periods=10, adjust=False).mean().values
    ema_full = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    hma_raw = 2 * ema_half - ema_full
    hma_12h = pd.Series(hma_raw).ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # Align HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
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
    
    warmup = max(50, donchian_period)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(hma_12h_aligned[i]) or
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
        # Volume confirmation: Require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Get trend from 12h HMA
            hma_trend = hma_12h_aligned[i]
            
            # Long: Price breaks above Donchian high + price above 12h HMA (uptrend)
            if high[i] > donchian_high[i] and price > hma_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Price breaks below Donchian low + price below 12h HMA (downtrend)
            elif low[i] < donchian_low[i] and price < hma_trend:
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