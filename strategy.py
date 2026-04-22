#!/usr/bin/env python3
"""
Hypothesis: 12-hour price action around 1-day VWAP with volume confirmation and ADX trend filter.
Long when price crosses above VWAP with rising volume in an uptrend (ADX>25).
Short when price crosses below VWAP with rising volume in a downtrend (ADX>25).
Exit when price reverts to VWAP or trend weakens (ADX<20).
Designed for low trade frequency by requiring VWAP cross + volume spike + trend alignment.
Works in both bull and bear markets by following the intraday trend via ADX.
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
    
    # Calculate VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.divide(vwap_numerator, vwap_denominator, 
                     out=np.full_like(vwap_numerator, np.nan), 
                     where=vwap_denominator!=0)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr[i] = max(high[i] - low[i], 
                        abs(high[i] - close[i-1]), 
                        abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(tr)
        minus_di = np.zeros_like(tr)
        dx = np.zeros_like(tr)
        
        atr[period-1] = np.mean(tr[1:period]) if period > 1 else tr[0]
        plus_dm_sum = np.sum(plus_dm[1:period]) if period > 1 else plus_dm[0]
        minus_dm_sum = np.sum(minus_dm[1:period]) if period > 1 else minus_dm[0]
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum/period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum/period) + minus_dm[i]
            
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # ADX is smoothed DX
        adx = np.full_like(dx, np.nan)
        adx[2*period-2] = np.mean(dx[period:2*period-1]) if 2*period-1 <= len(dx) else np.nan
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(adx[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price relative to VWAP
        price_above_vwap = close[i] > vwap[i]
        price_below_vwap = close[i] < vwap[i]
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend conditions
        strong_uptrend = adx[i] > 25 and ema50_1d_aligned[i] > ema50_1d_aligned[i-1]
        strong_downtrend = adx[i] > 25 and ema50_1d_aligned[i] < ema50_1d_aligned[i-1]
        weak_trend = adx[i] < 20
        
        if position == 0:
            # Long: Price crosses above VWAP with volume spike in uptrend
            if price_above_vwap and vol_spike and strong_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below VWAP with volume spike in downtrend
            elif price_below_vwap and vol_spike and strong_downtrend:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price reverts to VWAP or trend weakens
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below VWAP or trend weakens
                if price_below_vwap or weak_trend:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above VWAP or trend weakens
                if price_above_vwap or weak_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_VWAP_Trend_Volume_ADX"
timeframe = "12h"
leverage = 1.0