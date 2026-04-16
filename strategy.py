#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX Regime Filter
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (strong trend)
# Short when Bull Power < 0 AND Bear Power > 0 AND 1d ADX > 25
# Uses ATR-based trailing stop (2.5*ATR from extreme) and discrete position size 0.25
# Works in both bull/bear markets by requiring strong trend via ADX filter
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # === 1d Indicators: ADX for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = pd.Series(low_1d).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6h ATR for trailing stop ===
    tr1_6h = pd.Series(high).diff()
    tr2_6h = pd.Series(low).diff().abs()
    tr3_6h = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    extreme_price = 0.0  # highest high for long, lowest low for short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(atr_6h[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        high_val = high[i]
        low_val = low[i]
        adx_val = adx_aligned[i]
        atr_val = atr_6h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Update extreme price (highest high)
            extreme_price = max(extreme_price, high_val)
            # Trailing stop: 2.5*ATR below extreme
            if price < extreme_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Update extreme price (lowest low)
            extreme_price = min(extreme_price, low_val)
            # Trailing stop: 2.5*ATR above extreme
            if price > extreme_price + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            extreme_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Strong trend filter: ADX > 25
            if adx_val > 25.0:
                # LONG: Bull Power > 0 AND Bear Power < 0 (bullish momentum)
                if bull_power[i] > 0 and bear_power[i] < 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    extreme_price = high_val
                
                # SHORT: Bull Power < 0 AND Bear Power > 0 (bearish momentum)
                elif bull_power[i] < 0 and bear_power[i] > 0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    extreme_price = low_val
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dADX_RegimeFilter_V1"
timeframe = "6h"
leverage = 1.0