#!/usr/bin/env python3
"""
Experiment #039: 6h Williams %R(14) Extreme + 12h Volume Spike + 1d Trend Filter
HYPOTHESIS: Williams %R identifies overextended moves on 6h; volume spike confirms participation; 
12h EMA trend filter ensures alignment with intermediate trend. Works in bull/bear by taking 
extreme reversals in direction of 12h trend. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_039_6h_williamsr14_12h_volume_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for EMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(21) on 12h close
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.zeros(n)
    williams_r[:] = -100  # Initialize to -100 (oversold)
    # Avoid division by zero
    denominator = highest_high - lowest_low
    mask = denominator != 0
    williams_r[mask] = -100 * (highest_high[mask] - close[mask]) / denominator[mask]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = 50  # Warmup for indicator stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Williams %R Conditions ---
        williams_oversold = williams_r[i] < -80  # Oversold
        williams_overbought = williams_r[i] > -20  # Overbought
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- 12h EMA Trend: Determine intermediate trend direction ---
        price = close[i]
        ema_12h_trend_up = price > ema_12h_aligned[i]  # Above 12h EMA = uptrend
        ema_12h_trend_down = price < ema_12h_aligned[i]  # Below 12h EMA = downtrend
        
        # --- 1d EMA Trend Filter: Ensure alignment with higher timeframe trend ---
        ema_1d_trend_up = price > ema_1d_aligned[i]  # Above 1d EMA = bullish bias
        ema_1d_trend_down = price < ema_1d_aligned[i]  # Below 1d EMA = bearish bias
        
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
                # Exit on Williams %R extreme in opposite direction (mean reversion)
                if williams_overbought and volume_spike:
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
                # Exit on Williams %R extreme in opposite direction (mean reversion)
                if williams_oversold and volume_spike:
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
        # Long: Williams %R oversold + volume spike + price above 12h EMA + price above 1d EMA
        if williams_oversold and volume_spike and ema_12h_trend_up and ema_1d_trend_up:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Williams %R overbought + volume spike + price below 12h EMA + price below 1d EMA
        elif williams_overbought and volume_spike and ema_12h_trend_down and ema_1d_trend_down:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals