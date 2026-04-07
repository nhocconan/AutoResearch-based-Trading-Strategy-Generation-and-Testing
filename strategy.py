#!/usr/bin/env python3
"""
1h_triple_ema_trend_4h1d_filter_v1
Hypothesis: On 1h timeframe, use triple EMA (8,21,55) for trend direction and momentum, filtered by 4h and 1d EMA trends to avoid counter-trend trades. Enter long when short EMA > medium EMA > long EMA with bullish alignment on higher timeframes, and short when the reverse is true. Uses dynamic position sizing based on trend strength (ADX) to reduce whipsaws. Target: 60-150 trades over 4 years (15-37/year) with strict trend alignment filters to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_triple_ema_trend_4h1d_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate EMAs
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).values
    ema55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).values
    
    # Calculate ADX for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Load 4h EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 55:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).values
    ema55_4h = pd.Series(close_4h).ewm(span=55, adjust=False, min_periods=55).values
    
    # 4h trend: bullish if EMA21 > EMA55
    trend_4h_bullish = ema21_4h > ema55_4h
    trend_4h_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_21_4h := ema21_4h > ema55_4h)
    
    # Load 1d EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).values
    ema55_1d = pd.Series(close_1d).ewm(span=55, adjust=False, min_periods=55).values
    
    # 1d trend: bullish if EMA21 > EMA55
    trend_1d_bullish = ema21_1d > ema55_1d
    trend_1d_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_bullish)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(55, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(ema8[i]) or np.isnan(ema21[i]) or np.isnan(ema55[i]) or \
           np.isnan(adx[i]) or np.isnan(trend_4h_bullish_aligned[i]) or np.isnan(trend_1d_bullish_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine 1h EMA alignment
        ema_bullish = ema8[i] > ema21[i] > ema55[i]
        ema_bearish = ema8[i] < ema21[i] < ema55[i]
        
        # Trend strength filter: require ADX > 20
        strong_trend = adx[i] > 20
        
        if position == 1:  # Long position
            # Exit: EMA alignment breaks or trend weakens
            if not ema_bullish or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                # Scale position by trend strength (0.2 to 0.4 based on ADX)
                adx_factor = min(0.4, 0.2 + (adx[i] - 20) * 0.005)  # ADX 20-60 -> 0.2-0.4
                signals[i] = adx_factor
                
        elif position == -1:  # Short position
            # Exit: EMA alignment breaks or trend weakens
            if not ema_bearish or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                # Scale position by trend strength (0.2 to 0.4 based on ADX)
                adx_factor = min(0.4, 0.2 + (adx[i] - 20) * 0.005)  # ADX 20-60 -> 0.2-0.4
                signals[i] = -adx_factor
        else:  # Flat, look for entry
            # Only enter if higher timeframes align and strong trend
            if trend_4h_bullish_aligned[i] and trend_1d_bullish_aligned[i] and ema_bullish and strong_trend:
                position = 1
                adx_factor = min(0.4, 0.2 + (adx[i] - 20) * 0.005)
                signals[i] = adx_factor
            elif (not trend_4h_bullish_aligned[i]) and (not trend_1d_bullish_aligned[i]) and ema_bearish and strong_trend:
                position = -1
                adx_factor = min(0.4, 0.2 + (adx[i] - 20) * 0.005)
                signals[i] = -adx_factor
    
    return signals