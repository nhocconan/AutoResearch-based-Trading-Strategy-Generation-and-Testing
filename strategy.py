#!/usr/bin/env python3
"""
Experiment #212: 12h Donchian(20) breakout + 1d trend + volume confirmation + ATR stoploss

HYPOTHESIS: 12h Donchian(20) breakouts with volume confirmation (>1.5x average) and 
aligned 1-day trend (close > 1d EMA50) capture strong momentum moves. ATR-based 
stoploss (2.5x) manages risk. 12h timeframe targets 12-37 trades/year (50-150 total 
over 4 years) to minimize fee drag. Works in bull (breakouts with volume) and bear 
(failed reversals at Donchian levels) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_212_12h_donchian_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period=20):
        """Calculate Donchian channels: upper=max(high,period), lower=min(low,period)"""
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = calculate_donchian(high, low, 20)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr_12h = np.zeros(n)
    tr_12h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_12h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
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
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Price Levels ---
        price = close[i]
        upper = upper_20[i]
        lower = lower_20[i]
        ema50 = ema50_1d_aligned[i]
        
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
        # Long breakout: Price > Upper Donchian + volume spike + 1d trend up (price > EMA50)
        long_breakout = (price > upper) and volume_spike and (price > ema50)
        
        # Short breakout: Price < Lower Donchian + volume spike + 1d trend down (price < EMA50)
        short_breakout = (price < lower) and volume_spike and (price < ema50)
        
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