#/usr/bin/env python3
"""
6h_ElderRay_Energy_Index_With_WeeklyTrend_v1
Hypothesis: Elder Ray (Bull Power, Bear Power) combined with weekly trend filter and volume confirmation. 
In bull markets (weekly close > weekly SMA50): buy when Bull Power > 0 and rising with volume. 
In bear markets (weekly close < weekly SMA50): sell when Bear Power < 0 and falling with volume. 
Elder Ray measures bull/bear power behind price moves; weekly trend filters for major regime; 
volume confirms institutional participation. Designed for low frequency (12-30 trades/year) to 
minimize fee drag while capturing strong directional moves in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: SMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma50)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Smooth Elder Ray signals with 5-period EMA to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation: >1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirmed = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for weekly SMA50 and smoothing
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_sma50_aligned[i]) or
            np.isnan(bull_power_smooth[i]) or
            np.isnan(bear_power_smooth[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        weekly_trend_up = weekly_close.iloc[i] > weekly_sma50.iloc[i] if hasattr(weekly_close, 'iloc') else weekly_close[i] > weekly_sma50[i]
        weekly_trend_down = weekly_close.iloc[i] < weekly_sma50.iloc[i] if hasattr(weekly_close, 'iloc') else weekly_close[i] < weekly_sma50[i]
        bull_signal = bull_power_smooth[i] > 0 and bull_power_smooth[i] > bull_power_smooth[i-1]
        bear_signal = bear_power_smooth[i] < 0 and bear_power_smooth[i] < bear_power_smooth[i-1]
        vol_ok = volume_confirmed[i]
        
        if position == 0:
            # Long in uptrend: rising bull power with volume
            if weekly_trend_up and bull_signal and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short in downtrend: falling bear power with volume
            elif weekly_trend_down and bear_signal and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: bull power turns negative or weekly trend fails
            if bull_power_smooth[i] <= 0 or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: bear power turns positive or weekly trend fails
            if bear_power_smooth[i] >= 0 or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Energy_Index_With_WeeklyTrend_v1"
timeframe = "6h"
leverage = 1.0