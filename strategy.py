#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
Longs when price above Alligator teeth (green line) with 1d EMA50 uptrend and volume > 1.3x average.
Shorts when price below Alligator teeth with 1d EMA50 downtrend and volume > 1.3x average.
Exit when price crosses Alligator jaws (red line) or 1.5x ATR stop.
Designed for 15-25 trades/year to minimize fee dust while capturing sustained trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Alligator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs shifted into future
    # Jaw (blue): 13-period SMMA, shifted 8 bars
    # Teeth (green): 8-period SMMA, shifted 5 bars
    # Lips (red): 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift to align with Alligator logic (future values)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First values become NaN due to shift
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Alligator lines and EMA to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume spike > 1.3x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (14-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_50_val = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: price above teeth, EMA50 up, volume confirmation
            if (price_close > teeth_val and 
                ema_50_val > ema_50_val if i == 0 else ema_50_val > ema_50_aligned[i-1] and  # EMA50 rising
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: price below teeth, EMA50 down, volume confirmation
            elif (price_close < teeth_val and 
                  ema_50_val < ema_50_val if i == 0 else ema_50_val < ema_50_aligned[i-1] and  # EMA50 falling
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses jaws OR ATR-based stoploss
            exit_signal = False
            
            # Jaw crossover exit
            if position == 1 and price_close < jaw_val:
                exit_signal = True
            elif position == -1 and price_close > jaw_val:
                exit_signal = True
            
            # ATR-based stoploss (1.5x ATR from entry zone)
            if position == 1:
                # For longs, stop below lips (as proxy for recent support)
                if price_close < lips_val - 1.5 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above lips (as proxy for recent resistance)
                if price_close > lips_val + 1.5 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume1.3x_ATR1.5x"
timeframe = "12h"
leverage = 1.0