#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + ADX regime filter with volume confirmation.
- Uses Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure trend strength.
- Regime filter: ADX(14) > 25 for trending markets (only trade in strong trends).
- Volume confirmation: breakout requires volume > 1.5x 20-bar average.
- Long when Bull Power > 0 and ADX > 25 and volume confirms.
- Short when Bear Power < 0 and ADX > 25 and volume confirms.
- Uses 1d EMA34 as additional trend filter: price must be above/below 1d EMA34.
- Designed for 6h timeframe to capture medium-term trends with low trade frequency.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
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
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray components (Bull/Bear Power) on 6h timeframe
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # Calculate ADX(14) on 6h timeframe
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def ma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] + (arr[i] - result[i-1]) / period
        return result
    
    atr = ma(tr, 14)
    plus_di = 100 * ma(plus_dm, 14) / atr
    minus_di = 100 * ma(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = ma(dx, 14)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20)  # Need enough for EMA, ADX, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms and ADX indicates trend
            if volume_confirm and adx[i] > 25:
                # Long: Bull Power positive AND price above 1d EMA34
                if bull_power[i] > 0 and close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power negative AND price below 1d EMA34
                elif bear_power[i] < 0 and close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR price crosses below 1d EMA34 OR ADX weakens
            if bull_power[i] <= 0 or close[i] < ema_34_1d_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive OR price crosses above 1d EMA34 OR ADX weakens
            if bear_power[i] >= 0 or close[i] > ema_34_1d_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0