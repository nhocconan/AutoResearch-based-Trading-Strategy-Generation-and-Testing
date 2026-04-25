#!/usr/bin/env python3
"""
6h_WeeklyDonchian_Breakout_1dTrend_VolumeConfirmation
Hypothesis: 6-hour Donchian(20) breakout aligned with weekly trend (price > weekly EMA50) and volume confirmation (>1.6x 20-period average).
Weekly Donchian provides robust structure less prone to whipsaw in crypto.
1-day EMA50 filter ensures alignment with intermediate trend.
Volume confirmation avoids low-conviction breakouts.
Designed for ~80-160 total trades over 4 years (20-40/year) via multi-timeframe confluence.
Works in bull markets via trend-following breakouts and bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and Donchian calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need 50 for EMA50 and 20 for Donchian
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Get daily data for 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly Donchian channels (20-period)
    period_donchian = 20
    max_high_20w = pd.Series(high_1w).rolling(window=period_donchian, min_periods=period_donchian).max().values
    min_low_20w = pd.Series(low_1w).rolling(window=period_donchian, min_periods=period_donchian).min().values
    donchian_high = max_high_20w
    donchian_low = min_low_20w
    
    # Align weekly Donchian to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.6x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.6 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        weekly_trend = ema_50_1w_aligned[i]
        daily_trend = ema_50_1d_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        
        if position == 0:
            # Long conditions: price above weekly EMA50 (uptrend) AND breaks above weekly Donchian high with volume
            if close[i] > weekly_trend and close[i] > daily_trend:
                long_signal = (close[i] > donchian_high_val) and vol_regime[i]
            # Short conditions: price below weekly EMA50 (downtrend) AND breaks below weekly Donchian low with volume
            elif close[i] < weekly_trend and close[i] < daily_trend:
                short_signal = (close[i] < donchian_low_val) and vol_regime[i]
            else:
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (2.0*ATR from extreme)
            atr_stop = long_extreme - 2.0 * atr[i]
            # 2. Price breaks below weekly Donchian low (opposite boundary)
            if close[i] <= atr_stop or close[i] < donchian_low_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.0*ATR from extreme)
            atr_stop = short_extreme + 2.0 * atr[i]
            # 2. Price breaks above weekly Donchian high (opposite boundary)
            if close[i] >= atr_stop or close[i] > donchian_high_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchian_Breakout_1dTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0