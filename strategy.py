#!/usr/bin/env python3
"""
4h_ADX_WilliamsAlligator_Trend_V1
Hypothesis: 4h Williams Alligator (JAW/TEETH/LIPS) trend filter combined with ADX(14) > 25 for strong trend confirmation. Enter long when LIPS > TEETH > JAW and ADX > 25, short when LIPS < TEETH < JAW and ADX > 25. Uses 1d HTF EMA50 as additional trend filter (price > EMA50 for longs, < EMA50 for shorts). ATR-based stoploss via signal=0 when price moves against position by 2.5*ATR. Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag and work in both bull/bear markets via strong trend filtering. Focus on BTC/ETH.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Williams Alligator (primary timeframe) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Alligator lines: JAW (13,8), TEETH (8,5), LIPS (5,3) - all SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(np.median([high, low], axis=0), 13)  # 13-period SMMA of median price
    jaw = align_htf_to_ltf(prices, prices, jaw, additional_delay_bars=8)  # 8-bar delay for 13,5,3 alignment
    
    teeth = smma(np.median([high, low], axis=0), 8)   # 8-period SMMA of median price
    teeth = align_htf_to_ltf(prices, prices, teeth, additional_delay_bars=5)  # 5-bar delay
    
    lips = smma(np.median([high, low], axis=0), 5)    # 5-period SMMA of median price
    lips = align_htf_to_ltf(prices, prices, lips, additional_delay_bars=3)  # 3-bar delay
    
    # === ADX(14) for trend strength ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Wilder's smoothing: today = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # === ATR for stoploss ===
    atr_stop = atr  # Reuse ATR calculated above
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) 
            or np.isnan(adx[i]) or np.isnan(atr_stop[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        # Alligator alignment
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # ADX trend strength
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: Lips > Teeth > Jaw + strong trend + price > 1d EMA50
            if lips_above_teeth and teeth_above_jaw and strong_trend and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Lips < Teeth < Jaw + strong trend + price < 1d EMA50
            elif lips_below_teeth and teeth_below_jaw and strong_trend and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr_stop[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Alligator reverses or trend weakens
            elif not (lips_above_teeth and teeth_above_jaw) or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr_stop[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Alligator reverses or trend weakens
            elif not (lips_below_teeth and teeth_below_jaw) or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX_WilliamsAlligator_Trend_V1"
timeframe = "4h"
leverage = 1.0