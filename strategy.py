#!/usr/bin/env python3
"""
Experiment #350: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Daily Donchian breakouts above the 20-day high/low capture strong momentum moves.
Filtered by 1-week HMA21 trend to ensure alignment with higher timeframe direction.
Volume confirmation (>1.5x average) reduces false breakouts.
ATR-based stoploss (2.5x ATR) manages risk.
Designed for 1d timeframe targeting 30-100 total trades over 4 years (7-25/year).
Works in bull markets (breakouts with volume) and bear markets (failed reversals at Donchian levels with volume).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_350_1d_donchian_1w_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA21 trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA21 on 1w close
    def calculate_hma(arr, period):
        """Hull Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = arr.rolling(window=period//2, min_periods=period//2).mean()
        full = arr.rolling(window=period, min_periods=period).mean()
        raw_hma = 2.0 * half - full
        hma = raw_hma.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
        return hma.values
    
    hma21_1w = calculate_hma(pd.Series(df_1w['close']), 21)
    hma21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma21_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(hma21_1w_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Price Levels ---
        price = close[i]
        upper = donchian_h[i]
        lower = donchian_l[i]
        hma_trend = hma21_1w_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
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
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long breakout: Price > Upper Donchian + volume spike + 1w trend up (price > HMA)
        long_breakout = (price > upper) and volume_spike and (price > hma_trend)
        
        # Short breakout: Price < Lower Donchian + volume spike + 1w trend down (price < HMA)
        short_breakout = (price < lower) and volume_spike and (price < hma_trend)
        
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals