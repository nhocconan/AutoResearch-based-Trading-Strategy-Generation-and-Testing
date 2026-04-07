#!/usr/bin/env python3
"""
1h_adaptive_trend_follow_v1
Hypothesis: On 1h timeframe, follow 4h trend using ADX and EMA crossover, with 1d trend filter to avoid counter-trend trades. Uses volume confirmation and session filter (08-20 UTC) to reduce noise. Target 15-37 trades/year by requiring multiple confluence factors.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_adaptive_trend_follow_v1"
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
    volume = prices['volume'].values
    
    # Calculate 1h EMA crossover (8/21)
    ema_8 = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Calculate 1h ADX (14-period) for trend strength
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    # +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    plus_di_14 = wilders_smoothing(plus_dm, 14)
    minus_di_14 = wilders_smoothing(minus_dm, 14)
    
    # DI and ADX
    plus_di = np.where(tr_14 != 0, plus_di_14 / tr_14 * 100, 0)
    minus_di = np.where(tr_14 != 0, minus_di_14 / tr_14 * 100, 0)
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24h average
    
    # Calculate 4h EMA trend (21-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 1d trend filter (close vs 50 EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_8[i]) or np.isnan(ema_21[i]) or np.isnan(adx[i]) or 
            np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma[i]
        
        # Trend conditions
        ema_bullish = ema_8[i] > ema_21[i]
        ema_bearish = ema_8[i] < ema_21[i]
        adx_strong = adx[i] > 25
        
        # Multi-timeframe alignment
        trend_4h_up = close[i] > ema_21_4h_aligned[i]
        trend_4h_down = close[i] < ema_21_4h_aligned[i]
        trend_1d_up = close[i] > ema_50_1d_aligned[i]
        trend_1d_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: trend breaks or ADX weak
            if not (ema_bullish and adx_strong and trend_4h_up and trend_1d_up):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend breaks or ADX weak
            if not (ema_bearish and adx_strong and trend_4h_down and trend_1d_down):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if in_session and vol_ok and adx_strong:
                # Long: EMA bullish + 4h/1d uptrend
                if ema_bullish and trend_4h_up and trend_1d_up:
                    position = 1
                    signals[i] = 0.20
                # Short: EMA bearish + 4h/1d downtrend
                elif ema_bearish and trend_4h_down and trend_1d_down:
                    position = -1
                    signals[i] = -0.20
    
    return signals