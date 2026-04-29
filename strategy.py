#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX Regime
# Bull Power = High - EMA13(1d), Bear Power = EMA13(1d) - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (strong trend)
# Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (strong trend)
# Exit when Elder Ray signals weaken or ADX < 20 (trend weakening)
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years).
# Works in bull markets by capturing strong uptrends and in bear markets by capturing strong downtrends.
# ADX regime filter prevents whipsaws in ranging markets while allowing strong trends to run.

name = "6h_ElderRay_1dADX_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Elder Ray and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_1d = high_1d - ema_13_1d  # Bull Power = High - EMA13
    bear_power_1d = ema_13_1d - low_1d   # Bear Power = EMA13 - Low
    
    # Calculate 1d ADX for trend strength filter
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]  # handle first bar
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1]/period) + values[i]
        return result
    
    atr_period = 14
    tr_smoothed = wilders_smoothing(tr_1d, atr_period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, atr_period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, atr_period)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = wilders_smoothing(dx, atr_period)  # ADX is smoothed DX
    
    # Align all 1d indicators to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # ADX warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        adx = adx_1d_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Elder Ray weakens OR ADX < 20 (trend weakening)
            if bull_power <= 0 or bear_power >= 0 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Elder Ray weakens OR ADX < 20 (trend weakening)
            if bear_power <= 0 or bull_power >= 0 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (strong uptrend)
            if bull_power > 0 and bear_power < 0 and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power > 0 AND Bull Power < 0 AND ADX > 25 (strong downtrend)
            elif bear_power > 0 and bull_power < 0 and adx > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals