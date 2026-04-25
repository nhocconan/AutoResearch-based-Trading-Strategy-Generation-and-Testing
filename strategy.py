#!/usr/bin/env python3
"""
6h_ADX_Alligator_Confluence_v1
Hypothesis: Combine 6h ADX trend strength with 1d Williams Alligator for high-probability trend-following entries.
- Go long when: ADX(14) > 25 (strong trend) + price > Alligator Jaw (13-period SMMA) + Alligator Teeth (8-period SMMA) > Alligator Lips (5-period SMMA)
- Go short when: ADX(14) > 25 + price < Alligator Jaw + Alligator Teeth < Alligator Lips
- Exit when ADX < 20 (weakening trend) or Alligator lines cross in opposite direction
- Uses 1d Alligator for higher timeframe trend alignment, reducing false signals
- Target: 12-25 trades/year to stay under 300-trade 6h hard max
- Works in bull (ADX up + bullish Alligator alignment) and bear (ADX up + bearish Alligator alignment) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's MA or RSI-style smoothing"""
    if len(source) < period:
        return np.full(len(source), np.nan)
    result = np.full(len(source), np.nan)
    # First value is simple average
    result[period-1] = np.mean(source[:period])
    # Subsequent values: (prev_smma * (period-1) + current_price) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF Alligator indicator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for Alligator
        return np.zeros(n)
    
    # Calculate 6h ADX(14) for trend strength
    # ADX requires +DI, -DI, and DX calculations
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
        atr = np.full(len(tr), np.nan)
        plus_dm_smooth = np.full(len(tr), np.nan)
        minus_dm_smooth = np.full(len(tr), np.nan)
        
        # First values: simple average
        atr[period] = np.nanmean(tr[1:period+1])
        plus_dm_smooth[period] = np.nanmean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nanmean(minus_dm[1:period+1])
        
        # Subsequent values: Wilder's smoothing
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full(len(tr), np.nan)
        minus_di = np.full(len(tr), np.nan)
        dx = np.full(len(tr), np.nan)
        
        for i in range(period, len(tr)):
            if atr[i] != 0:
                plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # ADX: smoothed DX
        adx = np.full(len(tr), np.nan)
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(tr)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_6h = calculate_adx(high, low, close, 14)
    
    # Calculate 1d Williams Alligator (SMMA-based)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    close_1d = df_1d['close'].values
    jaw_1d = smma(close_1d, 13)
    teeth_1d = smma(close_1d, 8)
    lips_1d = smma(close_1d, 5)
    
    # Align Alligator lines to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ADX (~35 bars) and Alligator
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_6h[i]) or 
            np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine Alligator alignment
        alligator_bullish = (teeth_1d_aligned[i] > lips_1d_aligned[i]) and (jaw_1d_aligned[i] < teeth_1d_aligned[i])
        alligator_bearish = (teeth_1d_aligned[i] < lips_1d_aligned[i]) and (jaw_1d_aligned[i] > teeth_1d_aligned[i])
        
        # ADX trend strength filter
        strong_trend = adx_6h[i] > 25
        weakening_trend = adx_6h[i] < 20
        
        if position == 0:
            # Long setup: strong trend + bullish Alligator alignment + price above Jaw
            long_setup = strong_trend and alligator_bullish and (close[i] > jaw_1d_aligned[i])
            
            # Short setup: strong trend + bearish Alligator alignment + price below Jaw
            short_setup = strong_trend and alligator_bearish and (close[i] < jaw_1d_aligned[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: weakening trend OR Alligator turns bearish OR price crosses below Jaw
            if weakening_trend or (not alligator_bullish) or (close[i] <= jaw_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: weakening trend OR Alligator turns bullish OR price crosses above Jaw
            if weakening_trend or (not alligator_bearish) or (close[i] >= jaw_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Alligator_Confluence_v1"
timeframe = "6h"
leverage = 1.0