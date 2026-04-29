#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(9) signal line crossover with 1d ADX trend filter and volume confirmation
# Long when TRIX crosses above zero AND 1d ADX > 25 AND volume > 1.8x 20-bar average
# Short when TRIX crosses below zero AND 1d ADX > 25 AND volume > 1.8x 20-bar average
# Exit when TRIX crosses zero in opposite direction
# Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
# Target: 12-25 trades/year on 12h timeframe (48-100 total over 4 years) to avoid overtrading.
# TRIX is effective at catching momentum shifts in both bull and bear markets.
# ADX filter ensures we only trade in trending conditions, avoiding choppy markets.
# Volume confirmation adds conviction to breakouts.
# Works in bull markets by capturing upside momentum and in bear markets by capturing downside momentum
# with trend alignment preventing counter-trend trades.

name = "12h_TRIX_ADX_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) for trend filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/period)
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        
        # Calculate DI+ and DI-
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # Calculate DX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        dx[di_plus + di_minus == 0] = 0  # Avoid division by zero
        
        # Calculate ADX (smoothed DX)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate TRIX(9,9,9) - triple EMA of ROC
    def calculate_trix(close, period=9):
        # ROC = (Close - Close.period) / Close.period * 100
        roc = np.zeros_like(close)
        roc[period:] = (close[period:] - close[:-period]) / close[:-period] * 100
        
        # Triple EMA of ROC
        ema1 = pd.Series(roc).ewm(span=period, adjust=False, min_periods=period).mean().values
        ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
        ema3 = pd.Series(ema2).ewm(span=period, adjust=False, min_periods=period).mean().values
        return ema3
    
    trix = calculate_trix(close, 9)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 9*3)  # volume MA and TRIX warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_trix = trix[i]
        curr_adx = adx_1d_aligned[i]
        prev_trix = trix[i-1]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero
            if curr_trix < 0 and prev_trix >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero
            if curr_trix > 0 and prev_trix <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when TRIX crosses above zero AND ADX > 25 AND volume confirmation
            if curr_trix > 0 and prev_trix <= 0 and curr_adx > 25 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when TRIX crosses below zero AND ADX > 25 AND volume confirmation
            elif curr_trix < 0 and prev_trix >= 0 and curr_adx > 25 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals