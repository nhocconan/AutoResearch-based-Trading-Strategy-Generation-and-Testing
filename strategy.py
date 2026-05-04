#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX Regime + Volume Confirmation
# Elder Ray measures bull/bear power relative to EMA13. Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# In strong trends (ADX > 25): go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising.
# In ranging markets (ADX < 20): fade extremes - short when Bull Power > 0.7*ATR, long when Bear Power > 0.7*ATR.
# Volume confirmation requires >1.5x 20-period EMA volume to avoid low-activity false signals.
# Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "6h_ElderRay_1dADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Prepend NaN for alignment (since we lost first element in calculations)
    adx_full = np.concatenate([[np.nan], adx])
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_full)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # 6h ATR(14) for regime thresholds
    tr_6h = np.maximum(np.abs(high[1:] - low[1:]), 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                  np.abs(low[1:] - close[:-1])))
    atr_14_6h = pd.Series(np.concatenate([[np.nan], tr_6h])).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_14_6h[i]) or np.isnan(vol_ema_20[i]) or np.isnan(ema_13[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        adx_val = adx_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        atr_val = atr_14_6h[i]
        
        if position == 0:
            # Regime-based entries
            if adx_val > 25:  # Trending regime
                # Long: Bull Power positive and rising (vs previous bar)
                if bull_val > 0 and i > 100 and bull_val > bull_power[i-1]:
                    if volume_confirm:
                        signals[i] = 0.25
                        position = 1
                # Short: Bear Power positive and rising (vs previous bar)
                elif bear_val > 0 and i > 100 and bear_val > bear_power[i-1]:
                    if volume_confirm:
                        signals[i] = -0.25
                        position = -1
            elif adx_val < 20:  # Ranging regime
                # Fade bull power extremes
                if bull_val > 0.7 * atr_val:
                    if volume_confirm:
                        signals[i] = -0.25
                        position = -1
                # Fade bear power extremes
                elif bear_val > 0.7 * atr_val:
                    if volume_confirm:
                        signals[i] = 0.25
                        position = 1
        elif position == 1:
            # Exit long: Bull Power turns negative OR ADX drops below 20 (trend weakening)
            if bull_val <= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns negative OR ADX drops below 20 (trend weakening)
            if bear_val <= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals