#!/usr/bin/env python3
"""
4h_WilliamsAlligator_ElderRay_TrendFilter
Hypothesis: Combine Williams Alligator (trend direction) with Elder Ray (bull/bear power) for high-probability trend entries. 
Go long when Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 (price above EMA13). 
Go short when jaws > teeth > lips (bearish alignment) AND Bear Power < 0 (price below EMA13). 
Use 1d ADX > 25 for trend strength filter and volume > 1.5x 20-period average for confirmation. 
Exit on opposing Alligator signal or ADX < 20 (trend weakening). 
Designed for low turnover (<30 trades/year) with strong trend capture in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.full_like(tr, np.nan)
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    adx_1d = np.full_like(tr, np.nan)
    
    if len(tr) >= period + 1:
        # Initial ATR
        atr[period] = np.nanmean(tr[1:period+1])
        plus_di[period] = 100 * np.nanmean(plus_dm[1:period+1]) / atr[period]
        minus_di[period] = 100 * np.nanmean(minus_dm[1:period+1]) / atr[period]
        
        # Wilder smoothing
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            plus_di[i] = (plus_di[i-1] * (period - 1) + plus_dm[i]) / period * 100 / atr[i]
            minus_di[i] = (minus_di[i-1] * (period - 1) + minus_dm[i]) / period * 100 / atr[i]
        
        # DX and ADX
        di_sum = plus_di + minus_di
        dx = np.where(di_sum != 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0)
        
        # ADX smoothing
        adx_1d[2*period] = np.nanmean(dx[period+1:2*period+1])
        for i in range(2*period + 1, len(dx)):
            adx_1d[i] = (adx_1d[i-1] * (period - 1) + dx[i]) / period
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams Alligator on 4h
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Smoothed Moving Average (SMMA) - similar to Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: SMMA = (PREV_SMMA*(N-1) + CURRENT_PRICE) / N
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period - 1) + arr[i]) / period
        return result
    
    jaws = smma(close_4h, 13)  # Blue line
    teeth = smma(close_4h, 8)  # Red line
    lips = smma(close_4h, 5)   # Green line
    
    # Align Alligator lines to 4h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_4h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Elder Ray: Bull Power and Bear Power
    ema13 = smma(close_4h, 13)  # EMA13 equivalent
    ema13_aligned = align_htf_to_ltf(prices, df_4h, ema13)
    bull_power = high - ema13_aligned
    bear_power = low - ema13_aligned
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period, 2*14)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment signals
        bullish_alignment = jaws_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        bearish_alignment = jaws_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        
        # Elder Ray signals
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend strength filter
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Long: Bullish Alligator + Bull Power > 0 + Strong trend + Volume
            if bullish_alignment and bull_power_positive and strong_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + Bear Power < 0 + Strong trend + Volume
            elif bearish_alignment and bear_power_negative and strong_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish Alligator OR Weak trend
            if bearish_alignment or weak_trend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish Alligator OR Weak trend
            if bullish_alignment or weak_trend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_TrendFilter"
timeframe = "4h"
leverage = 1.0