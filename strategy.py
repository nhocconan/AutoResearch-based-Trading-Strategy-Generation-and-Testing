#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1-week ADX trend filter with 1-day volatility breakout.
In trending markets (weekly ADX > 25), buy breakouts above 1-day high with volume confirmation.
In ranging markets (weekly ADX < 20), sell breakdowns below 1-day low with volume confirmation.
Volume must exceed 2.0x 24-period average to confirm breakout/breakdown strength.
Exit on opposite volatility breakout or 1.5x ATR trailing stop.
Designed for 15-40 trades/year (60-160 total over 4 years) to minimize fee fade while capturing volatility expansion moves.
Works in bull markets via trend-following breakouts and in bear markets via mean-reversion breakdowns at volatility extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 24 or len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1-week ADX for trend/range filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth(wilders, period):
        smoothed = np.zeros_like(wilders)
        smoothed[period-1] = np.mean(wilders[:period])
        for i in range(period, len(wilders)):
            smoothed[i] = (smoothed[i-1] * (period-1) + wilders[i]) / period
        return smoothed
    
    atr_1w = smooth(tr, 14)
    dm_plus_smooth = smooth(dm_plus, 14)
    dm_minus_smooth = smooth(dm_minus, 14)
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / np.where(atr_1w == 0, 1, atr_1w)
    di_minus = 100 * dm_minus_smooth / np.where(atr_1w == 0, 1, atr_1w)
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    
    # ADX
    adx = smooth(dx, 14)
    adx_1w = adx
    
    # Calculate 1-day volatility breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day high/low for breakout levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    breakout_high = prev_high_1d
    breakdown_low = prev_low_1d
    
    # Align 1w and 1d indicators to 4h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    breakout_high_aligned = align_htf_to_ltf(prices, df_1d, breakout_high)
    breakdown_low_aligned = align_htf_to_ltf(prices, df_1d, breakdown_low)
    
    # Volume confirmation (volume spike > 2.0x 24-period average)
    vol_ma_24 = pd.Series(prices['volume'].values).rolling(window=24, min_periods=24).mean().values
    vol_ratio = prices['volume'].values / vol_ma_24
    
    # ATR for trailing stop (24-period)
    tr1_4h = prices['high'].values - prices['low'].values
    tr2_4h = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3_4h = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2_4h[0] = tr1_4h[0]
    tr3_4h[0] = tr1_4h[0]
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(breakout_high_aligned[i]) or 
            np.isnan(breakdown_low_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        adx_val = adx_1w_aligned[i]
        breakout_level = breakout_high_aligned[i]
        breakdown_level = breakdown_low_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr_4h[i]
        
        if position == 0:
            # Enter long: breakout above previous day high in trending market (ADX > 25)
            if (price_high > breakout_level and 
                adx_val > 25 and 
                vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Enter short: breakdown below previous day low in ranging market (ADX < 20)
            elif (price_low < breakdown_level and 
                  adx_val < 20 and 
                  vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit: opposite volatility breakout OR ATR trailing stop
            exit_signal = False
            
            # Opposite volatility breakout exit
            if position == 1 and price_low < breakdown_level:
                exit_signal = True
            elif position == -1 and price_high > breakout_level:
                exit_signal = True
            
            # ATR trailing stop (1.5x ATR from extreme)
            if position == 1:
                # Track highest high since entry
                if not hasattr(generate_signals, 'high_since_entry'):
                    generate_signals.high_since_entry = 0
                if price_high > generate_signals.high_since_entry:
                    generate_signals.high_since_entry = price_high
                if price_close < generate_signals.high_since_entry - 1.5 * atr_val:
                    exit_signal = True
            elif position == -1:
                # Track lowest low since entry
                if not hasattr(generate_signals, 'low_since_entry'):
                    generate_signals.low_since_entry = float('inf')
                if price_low < generate_signals.low_since_entry:
                    generate_signals.low_since_entry = price_low
                if price_close > generate_signals.low_since_entry + 1.5 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                # Reset tracking variables
                if hasattr(generate_signals, 'high_since_entry'):
                    delattr(generate_signals, 'high_since_entry')
                if hasattr(generate_signals, 'low_since_entry'):
                    delattr(generate_signals, 'low_since_entry')
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_ADXTrendFilter_VolatilityBreakout_Volume_ATR"
timeframe = "4h"
leverage = 1.0