#!/usr/bin/env python3
"""
6h ADX + Volume Breakout with 1d Trend Filter
Hypothesis: Strong trending moves (ADX > 25) with volume confirmation (>1.5x 20-period average) 
and alignment with 1-day EMA34 trend capture sustained momentum in both bull and bear markets.
ADX filters out choppy/range-bound periods, reducing false breakouts. Volume confirms 
institutional participation. Works in bull markets via upward breaks and in bear markets 
via downward breaks. Moderate trade frequency due to ADX and volume filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        if plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        elif minus_dm[i] < plus_dm[i]:
            minus_dm[i] = 0
        else:
            plus_dm[i] = 0
            minus_dm[i] = 0
            
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(high)
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    
    atr[period] = np.nansum(tr[1:period+1]) / period
    plus_dm_sum = np.nansum(plus_dm[1:period+1])
    minus_dm_sum = np.nansum(minus_dm[1:period+1])
    
    for i in range(period+1, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / atr[i]
        minus_di[i] = 100 * minus_dm_sum / atr[i]
    
    dx = np.zeros_like(high)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx[plus_di + minus_di == 0] = 0
    
    adx = np.zeros_like(high)
    adx[2*period] = np.nansum(dx[period+1:2*period+1]) / period
    for i in range(2*period+1, len(high)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 6h ADX for trend strength
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for ADX and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        trend = ema34_1d_aligned[i]
        strong_trend = adx[i] > 25
        vol_ok = vol_confirm[i]
        
        if position == 0:
            # Enter long: strong uptrend + volume + price above 1d EMA
            if strong_trend and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: strong downtrend + volume + price below 1d EMA
            elif strong_trend and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend weakness or reversal
            if not (strong_trend and close[i] > trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend weakness or reversal
            if not (strong_trend and close[i] < trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Volume_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0