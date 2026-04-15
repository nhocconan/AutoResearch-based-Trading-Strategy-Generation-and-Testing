#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h Supertrend filter and volume spike
# Long when 1h EMA9 crosses above EMA21 + 4h Supertrend uptrend + volume > 2.0x 20-period avg
# Short when 1h EMA9 crosses below EMA21 + 4h Supertrend downtrend + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.20) to control drawdown and minimize fee drag.
# 4h Supertrend provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (2.0x) targets ~15-35 trades/year on 1h timeframe to avoid overtrading.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # === 4h Indicator: Supertrend (ATR=10, mult=3.0) ===
    def calculate_supertrend(high, low, close, atr_period=10, multiplier=3.0):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # ATR
        atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
        
        # Basic Upper and Lower Bands
        basic_ub = (high + low) / 2 + multiplier * atr
        basic_lb = (high + low) / 2 - multiplier * atr
        
        # Final Upper and Lower Bands
        final_ub = np.zeros_like(close)
        final_lb = np.zeros_like(close)
        final_ub[0] = basic_ub[0]
        final_lb[0] = basic_lb[0]
        
        for i in range(1, len(close)):
            if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
        
        # Supertrend
        supertrend = np.zeros_like(close)
        supertrend[0] = final_ub[0]
        direction = np.zeros_like(close)
        direction[0] = 1  # Start with uptrend
        
        for i in range(1, len(close)):
            if close[i] > final_ub[i-1]:
                direction[i] = 1
            elif close[i] < final_lb[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
            if direction[i] == 1:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
                
        return supertrend, direction
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    supertrend_4h, supertrend_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, 10, 3.0)
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    supertrend_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_direction_4h)
    
    # === 1h Indicators: EMA9 and EMA21 ===
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(21, 20) + 5  # EMA21 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or
            np.isnan(supertrend_4h_aligned[i]) or np.isnan(supertrend_direction_4h_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # EMA crossover detection
        ema9_prev = ema9[i-1]
        ema21_prev = ema21[i-1]
        ema9_curr = ema9[i]
        ema21_curr = ema21[i]
        
        bullish_cross = (ema9_prev <= ema21_prev) and (ema9_curr > ema21_curr)
        bearish_cross = (ema9_prev >= ema21_prev) and (ema9_curr < ema21_curr)
        
        # === LONG CONDITIONS ===
        # 1. Bullish EMA9/EMA21 crossover
        # 2. 4h Supertrend uptrend (direction = 1)
        # 3. Volume confirmation
        if bullish_cross and \
           (supertrend_direction_4h_aligned[i] == 1) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Bearish EMA9/EMA21 crossover
        # 2. 4h Supertrend downtrend (direction = -1)
        # 3. Volume confirmation
        elif bearish_cross and \
             (supertrend_direction_4h_aligned[i] == -1) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_EMA9_EMA21_4hSupertrend_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0