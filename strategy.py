#!/usr/bin/env python3
"""
Hypothesis: 4-hour ADX trend strength with 1-day EMA trend filter and volume confirmation.
Long when ADX > 25 (trending) and EMA50 > EMA200 (bullish) with volume spike.
Short when ADX > 25 and EMA50 < EMA200 (bearish) with volume spike.
Exit when ADX < 20 (range) or EMA trend reverses.
ADX filters for trending markets to avoid whipsaws in ranges, EMA provides directional bias,
volume confirms institutional participation. Designed for low trade frequency by requiring
strong trend + volume confirmation. Works in bull/bear by following EMA trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX calculation
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first bar
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
        def WilderSmooth(data, period):
            smoothed = np.full_like(data, np.nan, dtype=float)
            smoothed[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
            return smoothed
        
        tr_smooth = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / tr_smooth
        minus_di = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = WilderSmooth(dx, period)
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # EMA50 and EMA200 for trend direction
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Load 1d data for EMA trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 34-period EMA on 1d close for higher timeframe trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(ema50[i]) or np.isnan(ema200[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Trend conditions
        bullish_trend = ema50[i] > ema200[i]
        bearish_trend = ema50[i] < ema200[i]
        strong_trend = adx[i] > 25
        ranging = adx[i] < 20
        
        if position == 0:
            # Enter long: strong uptrend + volume spike
            if strong_trend and bullish_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: strong downtrend + volume spike
            elif strong_trend and bearish_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend weakens or reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: trend weakens or turns bearish
                if ranging or not bullish_trend or (ema34_1d_aligned[i] < ema34_1d_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: trend weakens or turns bullish
                if ranging or not bearish_trend or (ema34_1d_aligned[i] > ema34_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_ADX_EMATrend_Filter_Volume"
timeframe = "4h"
leverage = 1.0