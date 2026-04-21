#!/usr/bin/env python3
"""
1d_MultiTimeframe_Trend_Follow
Hypothesis: Follow multi-timeframe trend using 1w EMA200 and 1d ADX on daily timeframe. 
Long when weekly trend up (price > weekly EMA200) and daily ADX > 25 with +DI > -DI.
Short when weekly trend down (price < weekly EMA200) and daily ADX > 25 with -DI > +DI.
Uses daily ATR for volatility filter to avoid choppy markets.
Designed for 1d timeframe to target 10-20 trades/year with high-conviction entries.
Works in bull markets by capturing continuation and in bear markets by capturing breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
            
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed TR, +DM, -DM
    atr = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
        plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
        minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Directional Indicators
    plus_di = np.zeros_like(atr)
    minus_di = np.zeros_like(atr)
    dx = np.zeros_like(atr)
    
    for i in range(period, len(atr)):
        if atr[i] != 0:
            plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # ADX
    adx = np.zeros_like(dx)
    if len(dx) >= 2 * period - 1:
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
    
    for i in range(2*period-1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx, plus_di, minus_di

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for trend filter
    ema200_1w = calculate_ema(close_1w, 200)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Load daily data for ADX and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ADX for trend strength
    adx_1d, plus_di_1d, minus_di_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d)
    
    # Daily ATR for volatility filter
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(plus_di_1d_aligned[i]) or np.isnan(minus_di_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        if i >= 50:
            vol_filter = atr_1d_aligned[i] > np.percentile(atr_1d_aligned[:i+1], 30)
        else:
            vol_filter = True
        
        if position == 0:
            # Weekly uptrend: price > weekly EMA200
            if price > ema200_1w_aligned[i]:
                # Long: strong uptrend (ADX > 25 and +DI > -DI)
                if (adx_1d_aligned[i] > 25 and 
                    plus_di_1d_aligned[i] > minus_di_1d_aligned[i] and 
                    vol_filter):
                    signals[i] = 0.25
                    position = 1
            # Weekly downtrend: price < weekly EMA200
            elif price < ema200_1w_aligned[i]:
                # Short: strong downtrend (ADX > 25 and -DI > +DI)
                if (adx_1d_aligned[i] > 25 and 
                    minus_di_1d_aligned[i] > plus_di_1d_aligned[i] and 
                    vol_filter):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: weekly trend reversal or weak trend
            if (price < ema200_1w_aligned[i] or 
                adx_1d_aligned[i] < 20 or 
                not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend reversal or weak trend
            if (price > ema200_1w_aligned[i] or 
                adx_1d_aligned[i] < 20 or 
                not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_MultiTimeframe_Trend_Follow"
timeframe = "1d"
leverage = 1.0