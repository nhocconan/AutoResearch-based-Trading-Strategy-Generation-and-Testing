#!/usr/bin/env python3
# 6h_williams_alligator_adx_regime_v1
# Hypothesis: 6h strategy combining Williams Alligator (trend direction) with ADX regime filter.
# Long when price > Alligator Jaw (13-period SMMA) and ADX > 25 (strong trend).
# Short when price < Alligator Jaw and ADX > 25.
# Exit when price crosses Alligator Jaw or ADX < 20 (weak trend/ranging).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 15-30 trades/year (60-120 total over 4 years) on BTC/ETH/SOL.
# Works in both bull and bear markets: Alligator identifies trend direction, ADX filters whipsaws in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williams_alligator_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator: three smoothed moving averages
    # Jaw (Blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (Red): 8-period SMMA, shifted 5 bars ahead
    # Lips (Green): 5-period SMMA, shifted 3 bars ahead
    close_s = pd.Series(close)
    
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = series.rolling(window=period, min_periods=period).mean()
        # Initialize first value as SMA
        result = np.full(len(series), np.nan)
        result[period-1] = sma.iloc[period-1]
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(series)):
            if not np.isnan(sma.iloc[i]):
                result[i] = (result[i-1] * (period-1) + series.iloc[i]) / period
        return result
    
    jaw = smma(close_s, 13)  # Jaw: 13-period
    teeth = smma(close_s, 8)  # Teeth: 8-period
    lips = smma(close_s, 5)   # Lips: 5-period
    
    # Shift as per Alligator specification
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # ADX calculation (14-period)
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = pd.Series(high - np.roll(high, 1))
    down_move = pd.Series(np.roll(low, 1) - low)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr_period = 14
    tr_s = pd.Series(tr)
    plus_dm_s = pd.Series(plus_dm)
    minus_dm_s = pd.Series(minus_dm)
    
    atr = tr_s.ewm(alpha=1/atr_period, min_periods=atr_period, adjust=False).mean().values
    plus_dm_smooth = plus_dm_s.ewm(alpha=1/atr_period, min_periods=atr_period, adjust=False).mean().values
    minus_dm_smooth = minus_dm_s.ewm(alpha=1/atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(atr != 0, atr, 1e-10)
    minus_di = 100 * minus_dm_smooth / np.where(atr != 0, atr, 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/atr_period, min_periods=atr_period, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(adx[i]) or 
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Jaw OR ADX < 20 (weakening trend)
            if close[i] < jaw_shifted[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Jaw OR ADX < 20 (weakening trend)
            if close[i] > jaw_shifted[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter: price > Jaw AND ADX > 25 (strong trend) for long
            #          price < Jaw AND ADX > 25 (strong trend) for short
            if close[i] > jaw_shifted[i] and adx[i] > 25:
                position = 1
                signals[i] = 0.25
            elif close[i] < jaw_shifted[i] and adx[i] > 25:
                position = -1
                signals[i] = -0.25
    
    return signals