#!/usr/bin/env python3
"""
Experiment #353: 4h Donchian Breakout + 12h HMA Trend + Volume Spike

HYPOTHESIS: Donchian channel breakouts on 4h capture strong momentum moves. 
Confirmed by 12h HMA trend alignment (HMA21) and volume spike (>2x average). 
ATR-based stoploss (2.5x) manages risk. 4h timeframe targets 20-50 trades/year 
(80-200 total over 4 years) to minimize fee drag. Works in bull markets (breakouts 
with volume) and bear markets (failed reversals at channel edges). 
Volume confirmation and trend filter reduce false breakouts in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_353_4h_donchian_12h_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA (Hull Moving Average) for 12h
    def calculate_hma(arr, period):
        """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
        half_period = max(1, period // 2)
        sqrt_period = max(1, int(np.sqrt(period)))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1, dtype=np.float64)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        arr_pd = pd.Series(arr)
        wma_half = wma(arr_pd.values, half_period)
        wma_full = wma(arr_pd.values, period)
        
        # Align arrays (WMA produces fewer values)
        raw = 2 * wma_half[-len(wma_full):] - wma_full
        hma = wma(raw, sqrt_period)
        
        # Pad to original length
        hma_padded = np.full(len(arr), np.nan)
        hma_padded[-len(hma):] = hma
        return hma_padded
    
    # Calculate HMA21 for 12h
    hma21_12h = calculate_hma(df_12h['close'].values, 21)
    hma21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma21_12h)
    
    # === 4h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period):
        """Donchian Channel: upper=max(high,period), lower=min(low,period)"""
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = 100  # Warmup for Donchian and volume MA stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma21_12h_aligned[i]) or np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: 12h HMA21 slope (rising/falling) ---
        if i >= warmup + 1:
            hma_now = hma21_12h_aligned[i]
            hma_prev = hma21_12h_aligned[i-1]
            hma_rising = hma_now > hma_prev
            hma_falling = hma_now < hma_prev
        else:
            hma_rising = True  # Neutral for warmup
            hma_falling = True  # Neutral for warmup
        
        # --- Volume Confirmation: Require volume spike (> 2x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Price Levels ---
        price = close[i]
        upper = donch_upper[i]
        lower = donch_lower[i]
        
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
        # Long breakout: Price > Upper Donchian + volume spike + 12h HMA rising
        long_breakout = (price > upper) and volume_spike and hma_rising
        
        # Short breakout: Price < Lower Donchian + volume spike + 12h HMA falling
        short_breakout = (price < lower) and volume_spike and hma_falling
        
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