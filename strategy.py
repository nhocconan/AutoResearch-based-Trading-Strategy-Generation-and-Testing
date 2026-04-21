#!/usr/bin/env python3
"""
4h_Relative_Vigor_Index_12hTrend_Volume_Confirmation
Hypothesis: Use Relative Vigor Index (RVI) to detect momentum shifts with 12h trend filter and volume confirmation.
RVI compares closing price to trading range, smoothed to identify bullish/bearish momentum.
Designed to work in both bull and bear markets by following higher timeframe trend while using RVI for entry timing.
Target: 20-40 trades/year on 4h to avoid excessive trading and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h trend filter: 34-period EMA ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === Relative Vigor Index (RVI) on 4h ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_ = prices['open'].values
    
    # Calculate RVI numerator and denominator
    a = close - open_
    b = close - open_
    c = close - open_
    d = close - open_
    
    # Corrected RVI calculation
    num = (close - open_) + 2 * (close - open_) + 2 * (close - open_) + (close - open_)
    den = (high - low) + 2 * (high - low) + 2 * (high - low) + (high - low)
    
    # Simplify calculation
    num = (close - open_) + 2 * (close - open_) + 2 * (close - open_) + (close - open_)
    den = (high - low) + 2 * (high - low) + 2 * (high - low) + (high - low)
    
    # Proper RVI calculation
    num = (close - open_) + 2 * (close - open_) + 2 * (close - open_) + (close - open_)
    den = (high - low) + 2 * (high - low) + 2 * (high - low) + (high - low)
    
    # Actually compute correctly
    a = close - open_
    b = close - open_
    c = close - open_
    d = close - open_
    
    # RVI calculation per standard formula
    numerator = (close - open_) + 2 * (close - open_) + 2 * (close - open_) + (close - open_)
    denominator = (high - low) + 2 * (high - low) + 2 * (high - low) + (high - low)
    
    # Fix the calculation
    a = close - open_
    b = close - open_  # This is wrong, need to fix
    
    # Correct RVI calculation
    num = (close - open_) + 2 * (close - open_) + 2 * (close - open_) + (close - open_)
    den = (high - low) + 2 * (high - low) + 2 * (high - low) + (high - low)
    
    # Actually compute proper RVI
    price_change = close - open_
    price_range = high - low
    
    # RVI = SMA of (close - open) / (high - low) over 4 periods
    # But we need to weight recent prices more
    
    # Proper RVI calculation
    a = close - open_
    b = np.roll(close - open_, 1)
    c = np.roll(close - open_, 2)
    d = np.roll(close - open_, 3)
    
    num = a + 2 * b + 2 * c + d
    
    a_range = high - low
    b_range = np.roll(high - low, 1)
    c_range = np.roll(high - low, 2)
    d_range = np.roll(high - low, 3)
    
    den = a_range + 2 * b_range + 2 * c_range + d_range
    
    # Handle first values
    num[0:3] = 0
    den[0:3] = 1  # Avoid division by zero
    
    rvi_raw = np.where(den != 0, num / den, 0)
    
    # Smooth RVI with exponential moving average
    rvi = pd.Series(rvi_raw).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Signal line for RVI
    rvi_signal = pd.Series(rvi).ewm(span=4, adjust=False, min_periods=4).mean().values
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(rvi[i]) or
            np.isnan(rvi_signal[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_12h = ema_34_12h_aligned[i]
        rvi_val = rvi[i]
        rvi_signal_val = rvi_signal[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: RVI crosses above signal + volume spike + price above 12h EMA
            if (rvi_val > rvi_signal_val and 
                rvi[i-1] <= rvi_signal[i-1] and
                vol_spike > 1.4 and 
                price_close > trend_12h):
                signals[i] = 0.25
                position = 1
            # Short: RVI crosses below signal + volume spike + price below 12h EMA
            elif (rvi_val < rvi_signal_val and 
                  rvi[i-1] >= rvi_signal[i-1] and
                  vol_spike > 1.4 and 
                  price_close < trend_12h):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when RVI crosses signal in opposite direction
            if position == 1 and rvi_val < rvi_signal_val and rvi[i-1] >= rvi_signal[i-1]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and rvi_val > rvi_signal_val and rvi[i-1] <= rvi_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Relative_Vigor_Index_12hTrend_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0