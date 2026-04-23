#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Regime Filter (ADX + Chop).
Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND price > EMA21.
Short when Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 AND price < EMA21.
Exit when Elder Power signals reverse OR 1d ADX < 20 (range) OR price crosses EMA21 opposite.
Uses 1d HTF for regime to avoid whipsaws in low ADX markets. Elder Ray measures bull/bear strength via EMA13.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Elder Ray provides clear momentum, 1d ADX filters regime, EMA21 provides dynamic support/resistance.
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
    
    # Calculate 6h EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate 6h EMA21 for dynamic support/resistance
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d ADX for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+ , DM- with Welles Wilder's smoothing (alpha=1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_period = 14
    tr_smooth = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, (dm_plus_smooth / tr_smooth) * 100, 0)
    di_minus = np.where(tr_smooth != 0, (dm_minus_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, atr_period)  # ADX is smoothed DX
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d EMA21 for additional trend filter (optional)
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(21, 30)  # EMA21 needs 21, ADX needs ~30 for smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema21[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema21[i]
        adx_val = adx_aligned[i]
        
        # Elder Ray signals
        bull_signal = bull_power[i] > 0 and bear_power[i] < 0  # Bullish: HP > EMA13 > LP
        bear_signal = bear_power[i] < 0 and bull_power[i] > 0  # Bearish: LP < EMA13 < HP
        
        if position == 0:
            # Long: Bullish Elder Ray AND 1d ADX > 25 (trending) AND price > EMA21
            if bull_signal and adx_val > 25 and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Elder Ray AND 1d ADX > 25 AND price < EMA21
            elif bear_signal and adx_val > 25 and price < ema_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Elder Ray turns bearish OR ADX < 20 (range) OR price < EMA21
                if not bull_signal or adx_val < 20 or price < ema_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: Elder Ray turns bullish OR ADX < 20 OR price > EMA21
                if not bear_signal or adx_val < 20 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1dADX25_Regime_EMA21"
timeframe = "6h"
leverage = 1.0