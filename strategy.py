#!/usr/bin/env python3
"""
1h_Supertrend_Filtered_By_4h_ADX
Hypothesis: Supertrend(ATR=10,mult=3) on 1h for entry timing, filtered by 4h ADX > 25 for trend strength.
In bull markets: follow 1h long signals when 4h trend is strong.
In bear markets: follow 1h short signals when 4h trend is strong.
Uses 1h only for entry timing, 4h for trend filter to reduce whipsaw and false signals.
Target: 60-150 total trades over 4 years (15-37/year) with position size 0.20 to manage drawdown.
"""

name = "1h_Supertrend_Filtered_By_4h_ADX"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ADX(14)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Plus Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth with Wilder's smoothing (EMA with alpha=1/period)
        def wilder_smoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.nansum(data[:period]) / period
                for i in range(period, len(data)):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilder_smoothing(tr, period)
        plus_di = 100 * wilder_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilder_smoothing(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smoothing(dx, period)
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 1h Supertrend (ATR=10, multiplier=3)
    def calculate_supertrend(high, low, close, atr_period=10, multiplier=3):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # ATR
        atr = np.zeros_like(close)
        atr[:atr_period-1] = np.nan
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
        
        # Supertrend
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        supertrend = np.zeros_like(close)
        direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
        
        supertrend[:atr_period] = np.nan
        direction[:atr_period] = np.nan
        
        for i in range(atr_period, len(close)):
            if close[i] > upper_band[i-1]:
                direction[i] = 1
            elif close[i] < lower_band[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                    lower_band[i] = lower_band[i-1]
                if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                    upper_band[i] = upper_band[i-1]
            
            if direction[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
        
        return supertrend, direction
    
    supertrend_1h, trend_direction_1h = calculate_supertrend(high, low, close, 10, 3)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(supertrend_1h[i]) or 
            np.isnan(trend_direction_1h[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Supertrend uptrend AND 4h ADX > 25 (strong trend)
            if trend_direction_1h[i] == 1 and adx_4h_aligned[i] > 25:
                signals[i] = 0.20
                position = 1
            # Short: Supertrend downtrend AND 4h ADX > 25 (strong trend)
            elif trend_direction_1h[i] == -1 and adx_4h_aligned[i] > 25:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Supertrend downtrend OR ADX weak (< 20)
            if trend_direction_1h[i] == -1 or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Supertrend uptrend OR ADX weak (< 20)
            if trend_direction_1h[i] == 1 or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals