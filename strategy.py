#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams Alligator combination with 1d trend filter
# Elder Ray measures bull/bear power (EMA13-based), Alligator (Jaw/Teeth/Lips) identifies trend absence/presence.
# Long when bull power > 0, bear power < 0, Lips > Teeth > Jaw (bullish alignment), and price > 1d EMA50.
# Short when bear power > 0, bull power < 0, Lips < Teeth < Jaw (bearish alignment), and price < 1d EMA50.
# Uses discrete sizing (0.25) to limit fee drag. Target: 50-150 total trades over 4 years.

name = "6h_ElderRay_Alligator_1dTrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6h
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Williams Alligator on 6h: Jaw (SMMA13), Teeth (SMMA8), Lips (SMMA5)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)   # Red line
    lips = smma(close, 5)    # Green line
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(13, 50)  # Need 13 for Elder Ray, 50 for 1d EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bull power > 0, bear power < 0, bullish alignment, price > 1d EMA50
            if bull_power[i] > 0 and bear_power[i] < 0 and bullish_alignment and curr_close > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bear power > 0, bull power < 0, bearish alignment, price < 1d EMA50
            elif bear_power[i] > 0 and bull_power[i] < 0 and bearish_alignment and curr_close < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish alignment or price < 1d EMA50
            if bearish_alignment or curr_close < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish alignment or price > 1d EMA50
            if bullish_alignment or curr_close > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals