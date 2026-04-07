#!/usr/bin/env python3
"""
4h_adx_volume_momentum_v1
Hypothesis: In trending markets (ADX > 25), price momentum combined with volume confirmation provides
an edge for trend continuation. Enter long when price is above EMA20, ADX rising, and volume above average;
enter short when price below EMA20, ADX rising, and volume above average. Use 1d trend filter to avoid
counter-trend trades. Designed for 4h timeframe to target 20-50 trades/year, minimizing fee drag.
Works in both bull and bear markets by following the daily trend and using ADX to filter ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_adx_volume_momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA20 for momentum
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20).mean().values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr = np.zeros(len(high))
        atr[period] = np.nansum(tr[1:period+1]) / period
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        dx = np.zeros(len(high))
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = 100 * (plus_dm[i] / atr[i])
                minus_di[i] = 100 * (minus_dm[i] / atr[i])
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(len(high))
        adx[2*period-1] = np.nansum(dx[period:2*period]) / period
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_20[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema_50d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        adx_rising = adx[i] > adx[i-1] if i > 0 else False
        vol_confirmed = volume[i] > vol_ma[i]
        above_ema20 = close[i] > ema_20[i]
        below_ema20 = close[i] < ema_20[i]
        bullish_trend = ema_50d_aligned[i] > ema_50d_aligned[i-1] if i > 0 else False
        bearish_trend = ema_50d_aligned[i] < ema_50d_aligned[i-1] if i > 0 else False
        
        if position == 1:  # Long position
            # Exit: ADX falls below 25 or price crosses below EMA20
            if adx[i] < 25 or close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX falls below 25 or price crosses above EMA20
            if adx[i] < 25 or close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: ADX > 25 and rising, price above EMA20, volume confirmed, bullish daily trend
            if adx[i] > 25 and adx_rising and above_ema20 and vol_confirmed and bullish_trend:
                position = 1
                signals[i] = 0.25
            # Short: ADX > 25 and rising, price below EMA20, volume confirmed, bearish daily trend
            elif adx[i] > 25 and adx_rising and below_ema20 and vol_confirmed and bearish_trend:
                position = -1
                signals[i] = -0.25
    
    return signals