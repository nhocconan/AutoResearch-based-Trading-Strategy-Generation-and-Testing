#!/usr/bin/env python3
"""
6h Elder Ray + Volume Confirmation + 1d ADX Trend Filter
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures buying/selling pressure relative to trend. 
Combined with volume confirmation and 1d ADX > 25 for strong trends, this captures institutional momentum moves in both bull and bear markets. 
Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-30 trades/year on 6h.
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
    
    # Get 1d data for ADX trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength
    # ADX requires +DM, -DM, TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # first value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    adx_1d = adx
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA13, volume MA, and ADX
    start_idx = max(100, 20, 14+14)  # ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        adx_val = adx_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Bull Power > 0 (buying pressure) AND volume spike AND ADX > 25 (strong trend)
            long_entry = (curr_bull_power > 0) and vol_spike and (adx_val > 25)
            # Short: Bear Power > 0 (selling pressure) AND volume spike AND ADX > 25 (strong trend)
            short_entry = (curr_bear_power > 0) and vol_spike and (adx_val > 25)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Bull Power <= 0 (loss of buying pressure) OR ADX < 20 (trend weakening)
            if (curr_bull_power <= 0) or (adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Bear Power <= 0 (loss of selling pressure) OR ADX < 20 (trend weakening)
            if (curr_bear_power <= 0) or (adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_VolumeConfirm_1dADX25_Trend"
timeframe = "6h"
leverage = 1.0