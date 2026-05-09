#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Alligator_ElderRay_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly trend (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Get 1d data for Elder Ray and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Weekly EMA34 for trend
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # 1d Alligator: Jaw (TEETH13), Teeth (TEETH8), Lips (TEETH5) - all SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(df_1d['close'].values, 13)
    teeth = smma(df_1d['close'].values, 8)
    lips = smma(df_1d['close'].values, 5)
    
    # Align all to 6h
    ema34_1w_6h = align_htf_to_ltf(prices, df_1w, ema34_1w)
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1w_6h[i]) or np.isnan(bull_power_6h[i]) or 
            np.isnan(bear_power_6h[i]) or np.isnan(jaw_6h[i]) or 
            np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_trend = ema34_1w_6h[i]
        bull = bull_power_6h[i]
        bear = bear_power_6h[i]
        jaw_val = jaw_6h[i]
        teeth_val = teeth_6h[i]
        lips_val = lips_6h[i]
        
        # Alligator alignment: all three lines in order
        # Bullish: Lips > Teeth > Jaw
        # Bearish: Lips < Teeth < Jaw
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Long: Weekly uptrend + Bullish Alligator + Positive Bull Power
            if close[i] > weekly_trend and bullish_alignment and bull > 0:
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend + Bearish Alligator + Negative Bear Power
            elif close[i] < weekly_trend and bearish_alignment and bear < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Weekly trend reversal or Bearish Alligator or negative Bull Power
            if close[i] < weekly_trend or not bullish_alignment or bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Weekly trend reversal or Bullish Alligator or positive Bear Power
            if close[i] > weekly_trend or not bearish_alignment or bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals