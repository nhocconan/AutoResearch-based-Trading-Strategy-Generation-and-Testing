#!/usr/bin/env python3
"""
6h Elder Ray + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Elder Ray (Bull/Bear Power) captures institutional buying/selling pressure,
weekly pivot provides directional bias from higher timeframe, volume confirmation ensures
follow-through. Works in bull markets via buy-the-dips on Bull Power + weekly pivot support,
and in bear markets via sell-the-rallies on Bear Power + weekly pivot resistance.
Target: 12-37 trades/year on 6h timeframe.
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
    
    # Get 1w data for weekly pivot points (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low  # Resistance 1
    weekly_s1 = 2 * weekly_pivot - weekly_high  # Support 1
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate Elder Ray on 6h data: Bull Power = High - EMA13, Bear Power = Low - EMA13
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
        bull_power = high - ema_13  # Buying pressure
        bear_power = low - ema_13   # Selling pressure (negative values)
    else:
        ema_13 = np.full(n, np.nan)
        bull_power = np.full(n, np.nan)
        bear_power = np.full(n, np.nan)
    
    # Calculate ATR(14) for stoploss
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
    
    # Start index: need enough for EMA13, ATR, and volume MA to propagate
    start_idx = max(13, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema13_val = ema_13[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        pivot = weekly_pivot_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirmed = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) AND price above weekly pivot AND volume confirmed
            long_condition = (bull_val > 0) and (curr_close > pivot) and volume_confirmed
            # Short: Bear Power < 0 (selling pressure) AND price below weekly pivot AND volume confirmed
            short_condition = (bear_val < 0) and (curr_close < pivot) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or Bear Power turns negative (selling pressure)
            if curr_close <= entry_price - 2.0 * atr_val or bear_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or Bull Power turns positive (buying pressure)
            if curr_close >= entry_price + 2.0 * atr_val or bull_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_WeeklyPivot_Direction_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0