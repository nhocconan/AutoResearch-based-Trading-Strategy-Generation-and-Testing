#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_camarilla_breakout_v2
# Uses daily Camarilla pivot levels to identify key support/resistance.
# Long when price breaks above H4 resistance with volume confirmation.
# Short when price breaks below L4 support with volume confirmation.
# Uses 4h ADX > 20 to ensure trending market and avoid choppy conditions.
# Designed for moderate trade frequency (target: 20-50 trades/year) to balance opportunity and cost.
# Works in both bull and breakouts, and bear and breakdowns.

name = "4h_1d_camarilla_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla levels: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # Using previous day's range for today's levels
    camarilla_h4 = close_prev + 1.5 * (high_prev - low_prev)
    camarilla_l4 = close_prev - 1.5 * (high_prev - low_prev)
    
    # Align Camarilla levels to 4h timeframe (daily values update after daily bar closes)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # ADX filter on 4h to avoid choppy markets (ADX > 20 = trending)
    # Calculate directional movement
    high_diff = np.diff(high, prepend=high[0])
    low_diff = -np.diff(low, prepend=low[0])  # negative of low diff
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first period
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    adx_filter = adx > 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_confirm[i]) or np.isnan(adx_filter[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: price breaks above H4 with volume and trend
        if (close[i] > camarilla_h4_aligned[i] and vol_confirm[i] and adx_filter[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L4 with volume and trend
        elif (close[i] < camarilla_l4_aligned[i] and vol_confirm[i] and adx_filter[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to midpoint (Pivot) or opposite breakout
        elif position == 1 and close[i] < (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals