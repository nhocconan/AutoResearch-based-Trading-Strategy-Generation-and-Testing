#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1w EMA50 trend filter
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Williams Alligator identifies trend via jaw/teeth/lips alignment. Elder Ray measures bull/bear power.
# 1w EMA50 filter ensures trading only with the primary weekly trend to avoid counter-trend whipsaws.
# Works in both bull and bear markets by aligning with higher timeframe direction.

name = "6h_WilliamsAlligator_ElderRay_1wEMA50_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Williams Alligator (13,8,5 smoothed with 8,5,3)
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Calculate Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 1w EMA(50) for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 13, 8, 5, 13, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
            # Elder Ray confirmation: Bull Power > 0 and rising, Bear Power < 0 and falling
            # Weekly trend filter: Price > EMA50 for long, Price < EMA50 for short
            
            # Bullish entry: Alligator bullish alignment + Bull Power positive + above weekly EMA
            if (curr_lips > curr_teeth > curr_jaw and 
                curr_bull_power > 0 and 
                curr_close > curr_ema_50_1w):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Bearish entry: Alligator bearish alignment + Bear Power negative + below weekly EMA
            elif (curr_lips < curr_teeth < curr_jaw and 
                  curr_bear_power < 0 and 
                  curr_close < curr_ema_50_1w):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR Alligator loses bullish alignment OR Elder Ray turns negative
            if (curr_low <= stop_loss or 
                not (curr_lips > curr_teeth > curr_jaw) or 
                curr_bull_power <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR Alligator loses bearish alignment OR Elder Ray turns positive
            if (curr_high >= stop_loss or 
                not (curr_lips < curr_teeth < curr_jaw) or 
                curr_bear_power >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals