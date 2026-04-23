#!/usr/bin/env python3
"""
Hypothesis: 6h ADX + Williams Alligator combination with 1w trend filter
- Williams Alligator (Jaw/Teeth/Lips) identifies trend absence/presence via convergence/divergence
- ADX > 25 confirms strong trend, < 20 indicates ranging market
- Only trade Alligator signals in direction of 1w EMA(50) to avoid counter-trend whipsaws
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1w trend
- Alligator provides natural trend/filter system that adapts to market regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 6h data
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward  
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply forward shifts (Alligator lines are shifted into future)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that go beyond array bounds
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX (Average Directional Index)"""
        if len(high) < period + 1:
            return np.full(len(high), np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed TR, DM+
        def wilders_smoothing(data, period):
            """Wilder's smoothing (similar to EMA but with different alpha)"""
            if len(data) < period:
                return np.full(len(data), np.nan)
            result = np.full(len(data), np.nan)
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Wilder's smoothing: alpha = 1/period
            alpha = 1.0 / period
            for i in range(period, len(data)):
                if np.isnan(result[i-1]):
                    result[i] = np.nanmean(data[i-period+1:i+1])
                else:
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilders_smoothing(tr, period)
        dm_plus_smooth = wilders_smoothing(dm_plus, period)
        dm_minus_smooth = wilders_smoothing(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = wilders_smoothing(dx, period)
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 13) + 8  # EMA, ADX, plus max Alligator shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(adx[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Alligator state
        # Converging (all lines intertwined) = ranging/no trend
        # Diverging with proper order = trending
        lips_above_teeth = lips_shifted[i] > teeth_shifted[i]
        teeth_above_jaw = teeth_shifted[i] > jaw_shifted[i]
        lips_above_jaw = lips_shifted[i] > jaw_shifted[i]
        
        lips_below_teeth = lips_shifted[i] < teeth_shifted[i]
        teeth_below_jaw = teeth_shifted[i] < jaw_shifted[i]
        lips_below_jaw = lips_shifted[i] < jaw_shifted[i]
        
        # Bullish Alligator: Lips > Teeth > Jaw (all diverging upward)
        bullish_alligator = lips_above_teeth and teeth_above_jaw and lips_above_jaw
        
        # Bearish Alligator: Lips < Teeth < Jaw (all diverging downward)
        bearish_alligator = lips_below_teeth and teeth_below_jaw and lips_below_jaw
        
        # Trend filter from 1w EMA
        uptrend_1w = close[i] > ema_50_1w_aligned[i]
        downtrend_1w = close[i] < ema_50_1w_aligned[i]
        
        # ADX filter for trend strength
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20  # Ranging market
        
        if position == 0:
            # Long conditions: Bullish Alligator + uptrend + strong ADX
            long_signal = bullish_alligator and uptrend_1w and strong_trend
            
            # Short conditions: Bearish Alligator + downtrend + strong ADX
            short_signal = bearish_alligator and downtrend_1w and strong_trend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator convergence OR trend reversal OR weak ADX
            exit_signal = False
            
            # Alligator convergence (lines intertwining) = end of trend
            alligator_converging = (
                abs(lips_shifted[i] - teeth_shifted[i]) < (close[i] * 0.001) or  # Lips-Teeth close
                abs(teeth_shifted[i] - jaw_shifted[i]) < (close[i] * 0.001) or   # Teeth-Jaw close
                abs(lips_shifted[i] - jaw_shifted[i]) < (close[i] * 0.001)       # Lips-Jaw close
            )
            
            if position == 1:
                # Exit long: Bearish Alligator OR convergence OR trend reversal OR weak ADX
                if (bearish_alligator or 
                    alligator_converging or 
                    not uptrend_1w or 
                    weak_trend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Bullish Alligator OR convergence OR trend reversal OR weak ADX
                if (bullish_alligator or 
                    alligator_converging or 
                    not downtrend_1w or 
                    weak_trend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ADX_WilliamsAlligator_1wEMA50_TrendFilter"
timeframe = "6h"
leverage = 1.0