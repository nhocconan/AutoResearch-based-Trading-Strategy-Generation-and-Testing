#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures buying/selling pressure
# 1d ADX > 25 filters for trending markets only, avoiding whipsaws in ranging conditions
# Long when Bull Power > 0 and Bear Power < 0 in uptrend (ADX > 25 + EMA50 rising)
# Short when Bear Power > 0 and Bull Power < 0 in downtrend (ADX > 25 + EMA50 falling)
# Designed for ~15-25 trades/year to minimize fee drag while capturing strong trending moves
# Works in bull/bear via ADX regime filter - only trades when trend is strong

name = "6h_ElderRay_1dADX25_EMA50Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX and EMA50 regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (trend strength filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, period) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = wilders_smoothing(dx_1d, period)
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d regime filters to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h Elder Ray components
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13_6h  # Buying pressure
    bear_power = ema_13_6h - low   # Selling pressure
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13)  # Warmup for EMA50 and EMA13
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_adx = adx_1d_aligned[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Elder Ray turns bearish or trend weakens
            if curr_bull <= 0 or curr_bear >= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Elder Ray turns bullish or trend weakens
            if curr_bear <= 0 or curr_bull >= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Trend regime: ADX > 25 indicates strong trend
            strong_trend = curr_adx > 25
            
            # Long when Bull Power > 0 (buying pressure) and Bear Power < 0 (no selling pressure)
            # in uptrend (price above 1d EMA50)
            if (curr_bull > 0 and curr_bear < 0 and 
                strong_trend and curr_close > curr_ema50_1d):
                signals[i] = 0.25
                position = 1
            # Short when Bear Power > 0 (selling pressure) and Bull Power < 0 (no buying pressure)
            # in downtrend (price below 1d EMA50)
            elif (curr_bear > 0 and curr_bull < 0 and 
                  strong_trend and curr_close < curr_ema50_1d):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals