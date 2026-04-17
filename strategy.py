#!/usr/bin/env python3
"""
6h ADX + Williams Alligator with 12h Trend Filter
Long: ADX > 25 + Green > Mouth > Red (bullish) + price > 12h EMA50
Short: ADX > 25 + Red > Mouth > Green (bearish) + price < 12h EMA50
Exit: ADX < 20 or Alligator lines cross in opposite direction
Williams Alligator: Jaw (13 SMMA shifted 8), Teeth (8 SMMA shifted 5), Lips (5 SMMA shifted 3)
Uses 12h EMA50 to filter against higher timeframe trend, reducing false signals in chop
Target: 15-25 trades/year per symbol (60-100 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    sma = np.mean(series[:period])
    smma_vals = np.empty_like(series)
    smma_vals[:] = np.nan
    smma_vals[period-1] = sma
    for i in range(period, len(series)):
        smma_vals[i] = (smma_vals[i-1] * (period-1) + series[i]) / period
    return smma_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator components (6-period calculations)
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)  # Jaw: 13-period SMMA
    teeth = smma(median_price, 8)   # Teeth: 8-period SMMA
    lips = smma(median_price, 5)    # Lips: 5-period SMMA
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set rolled values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # ADX calculation (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder smoothing
    
    plus_di = 100 * np.where(atr > 0, 
                             np.convolve(plus_dm, np.ones(14)/14, mode='full')[:len(plus_dm)] / atr, 
                             0)
    minus_di = 100 * np.where(atr > 0,
                              np.convolve(minus_dm, np.ones(14)/14, mode='full')[:len(minus_dm)] / atr,
                              0)
    dx = 100 * np.where((plus_di + minus_di) > 0, 
                        np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                        0)
    adx = np.zeros_like(dx)
    adx[13] = np.mean(dx[14:28]) if len(dx) >= 28 else np.nan
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    # Set early values to NaN
    adx[:27] = np.nan
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, 30)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment
        bullish_alignment = (lips[i] > teeth[i] > jaw[i])
        bearish_alignment = (jaw[i] > teeth[i] > lips[i])
        
        if position == 0:
            # Long: ADX > 25 + bullish alignment + price > 12h EMA50
            if adx[i] > 25 and bullish_alignment and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 + bearish alignment + price < 12h EMA50
            elif adx[i] > 25 and bearish_alignment and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: ADX < 20 or bearish alignment
            if adx[i] < 20 or bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: ADX < 20 or bullish alignment
            if adx[i] < 20 or bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_WilliamsAlligator_12hEMA50"
timeframe = "6h"
leverage = 1.0