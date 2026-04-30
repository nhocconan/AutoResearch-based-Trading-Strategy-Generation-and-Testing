#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination with 12h trend filter
# ADX > 25 indicates strong trend (works in both bull/bear markets).
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) confirms trend direction and alignment.
# Long when ADX > 25 + Lips > Teeth > Jaw (bullish) + price above 12h EMA50.
# Short when ADX > 25 + Lips < Teeth < Jaw (bearish) + price below 12h EMA50.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 75-200 trades over 4 years.
# ADX filters out ranging markets, Alligator avoids whipsaws, 12h EMA50 provides HTF trend bias.

name = "6h_ADX_Alligator_12hEMA50_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 6h data
    # Jaw: 13-period SMMA (smoothed moving average) of median price
    # Teeth: 8-period SMMA of median price
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Calculate ADX (14-period) on 6h data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (similar to EMA with alpha=1/period)
        def wilder_smoothing(arr, period):
            result = np.full_like(arr, np.nan, dtype=float)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Wilder smoothing: today = (prev * (period-1) + current) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        tr_smoothed = wilder_smoothing(tr, period)
        plus_dm_smoothed = wilder_smoothing(plus_dm, period)
        minus_dm_smoothed = wilder_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smoothed / tr_smoothed
        minus_di = 100 * minus_dm_smoothed / tr_smoothed
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilder_smoothing(dx, period)
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(adx[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_adx = adx[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade when ADX > 25 (strong trend) with Alligator alignment and 12h trend filter
            if curr_adx > 25:
                # Bullish: Lips > Teeth > Jaw (bullish alignment) + price above 12h EMA50
                if curr_lips > curr_teeth > curr_jaw and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Lips < Teeth < Jaw (bearish alignment) + price below 12h EMA50
                elif curr_lips < curr_teeth < curr_jaw and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: ADX drops below 20 (trend weakening) OR Alligator loses bullish alignment
            if curr_adx < 20 or curr_lips <= curr_teeth or curr_teeth <= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: ADX drops below 20 (trend weakening) OR Alligator loses bearish alignment
            if curr_adx < 20 or curr_lips >= curr_teeth or curr_teeth >= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals