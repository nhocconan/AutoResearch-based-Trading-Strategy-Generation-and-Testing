#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation.
Long when price breaks above upper Bollinger Band AND 1d ADX > 25 AND volume > 1.5x 20-period average.
Short when price breaks below lower Bollinger Band AND 1d ADX > 25 AND volume > 1.5x 20-period average.
Exit when price retouches Bollinger middle band (20-period SMA) or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Bollinger Band squeeze (low volatility) precedes breakouts in both bull and bear markets.
ADX > 25 ensures we only trade breakouts in trending conditions, reducing false signals.
Targets 12-37 trades/year per symbol (50-150 total over 4 years) by requiring volatility contraction before expansion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate Bollinger Bands (20, 2) on 6h timeframe
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + (bb_std * bb_stddev)
    lower_bb = sma_bb - (bb_std * bb_stddev)
    middle_bb = sma_bb  # 20-period SMA for exit
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for ADX
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # first bar
    
    # Calculate +DM and -DM
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    adx_period = 14
    alpha = 1.0 / adx_period
    
    atr_1d = np.zeros_like(tr_1d)
    atr_1d[0] = tr_1d[0]
    for i in range(1, len(tr_1d)):
        atr_1d[i] = alpha * tr_1d[i] + (1 - alpha) * atr_1d[i-1]
    
    plus_di_1d = np.zeros_like(plus_dm)
    minus_di_1d = np.zeros_like(minus_dm)
    plus_di_1d[0] = 100 * plus_dm[0] / atr_1d[0] if atr_1d[0] != 0 else 0
    minus_di_1d[0] = 100 * minus_dm[0] / atr_1d[0] if atr_1d[0] != 0 else 0
    
    for i in range(1, len(plus_dm)):
        plus_di_1d[i] = alpha * (100 * plus_dm[i] / atr_1d[i] if atr_1d[i] != 0 else 0) + (1 - alpha) * plus_di_1d[i-1]
        minus_di_1d[i] = alpha * (100 * minus_dm[i] / atr_1d[i] if atr_1d[i] != 0 else 0) + (1 - alpha) * minus_di_1d[i-1]
    
    # Calculate DX and ADX
    dx_1d = np.zeros_like(plus_di_1d)
    for i in range(len(dx_1d)):
        di_sum = plus_di_1d[i] + minus_di_1d[i]
        dx_1d[i] = 100 * np.abs(plus_di_1d[i] - minus_di_1d[i]) / di_sum if di_sum != 0 else 0
    
    adx_1d = np.zeros_like(dx_1d)
    adx_1d[adx_period-1] = np.mean(dx_1d[:adx_period])  # First ADX value
    for i in range(adx_period, len(dx_1d)):
        adx_1d[i] = alpha * dx_1d[i] + (1 - alpha) * adx_1d[i-1]
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation on 6h timeframe
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 20, adx_period*2, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_bb[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        middle = middle_bb[i]
        adx_val = adx_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper BB AND ADX > 25 AND volume spike
            if (price > upper and 
                adx_val > 25 and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below lower BB AND ADX > 25 AND volume spike
            elif (price < lower and 
                  adx_val > 25 and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Bollinger middle band (20-period SMA)
            if position == 1 and price <= middle:
                exit_signal = True
            elif position == -1 and price >= middle:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_BollingerSqueeze_ADXTrend_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0