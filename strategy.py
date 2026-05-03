#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter + volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power rising AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period MA
# Short when Bear Power < 0 AND Bull Power falling AND 1d ADX > 25 AND volume > 1.5x 20-period MA
# Uses ATR-based stoploss (signal→0 when price moves against position by 2.5*ATR)
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year).
# Elder Ray measures bull/bear power relative to EMA13, filtering by 1d ADX ensures we only trade
# in strong trending markets (avoiding whipsaws in ranges), volume confirmation adds institutional participation.
# Works in both bull and bear markets by trading with the 1d trend direction via ADX regime.

name = "6h_ElderRay_1dADX25_VolumeSpike_ATR"
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
    
    # Calculate ATR for stoploss (using primary timeframe)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = 0
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for ATR-based stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        bull = bull_power[i]
        bear = bear_power[i]
        adx_val = adx_aligned[i]
        
        # Trend strength filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power rising (less negative) AND strong trend AND volume spike
            if bull > 0 and (i == 100 or bear > bear_power[i-1]) and strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Bear Power < 0 AND Bull Power falling (less positive) AND strong trend AND volume spike
            elif bear < 0 and (i == 100 or bull < bull_power[i-1]) and strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Stoploss: price moves against position by 2.5*ATR
            if close_val < entry_price - 2.5 * atr[i]:
                exit_signal = True
            # Exit: Bull Power becomes negative
            elif bull <= 0:
                exit_signal = True
            # Exit: Bear Power starts falling (more negative) indicating weakening bulls
            elif i > 100 and bear < bear_power[i-1]:
                exit_signal = True
            # Exit: Trend weakens (ADX < 20)
            elif adx_val < 20:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Stoploss: price moves against position by 2.5*ATR
            if close_val > entry_price + 2.5 * atr[i]:
                exit_signal = True
            # Exit: Bear Power becomes positive
            elif bear >= 0:
                exit_signal = True
            # Exit: Bull Power starts falling (more negative) indicating strengthening bears
            elif i > 100 and bull < bull_power[i-1]:
                exit_signal = True
            # Exit: Trend weakens (ADX < 20)
            elif adx_val < 20:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals