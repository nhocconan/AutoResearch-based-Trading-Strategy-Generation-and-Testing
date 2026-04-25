#!/usr/bin/env python3
"""
6h Weekly Pivot Fade + 1d EMA50 Trend + Volume Confirmation
Hypothesis: In ranging markets (2025+), weekly pivot levels (R2/S2) act as strong support/resistance.
Fade at these levels with trend filter (1d EMA50) and volume confirmation works in both bull/bear:
- Bull: Fade at S2 with uptrend (long)
- Bear: Fade at R2 with downtrend (short)
- Range: Fade both sides
Uses 6h timeframe to limit trades (target: 12-37/year) and reduce fee drag.
Weekly pivots calculated from prior week OHLC, aligned with 2-bar delay for confirmation.
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
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for weekly pivot points (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R2 = P + (H-L), S2 = P - (H-L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Williams fractal needs 2 extra 1d bars after the center bar for confirmation
    # But for weekly pivot, we use the completed weekly bar only (no extra delay needed)
    # align_htf_to_ltf() already waits for the weekly bar to close
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Calculate ATR(14) for stoploss on 6h data
    if len(close) >= 14:
        tr1 = np.abs(np.diff(close, prepend=close[0]))
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = np.abs(high[0] - close[0])
        tr3[0] = np.abs(low[0] - close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros(n)
        atr[:13] = np.nan
        for i in range(13, n):
            atr[i] = np.mean(tr[i-13:i+1])
    else:
        atr = np.full(n, np.nan)
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50_1d, ATR, and volume MA to propagate
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema50_1d = ema_50_1d_aligned[i]
        r2 = weekly_r2_aligned[i]
        s2 = weekly_s2_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: price at S2 (within 0.2% tolerance) AND uptrend (price > 1d EMA50) AND volume confirm
            long_condition = (abs(curr_close - s2) / s2 < 0.002) and (curr_close > ema50_1d) and volume_confirm
            # Short: price at R2 (within 0.2% tolerance) AND downtrend (price < 1d EMA50) AND volume confirm
            short_condition = (abs(curr_close - r2) / r2 < 0.002) and (curr_close < ema50_1d) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price reaches R2 (profit target)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close >= r2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price reaches S2 (profit target)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close <= s2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Fade_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0