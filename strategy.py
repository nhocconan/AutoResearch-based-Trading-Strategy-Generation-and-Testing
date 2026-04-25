#!/usr/bin/env python3
"""
6h_ADX_Trend_ElderRay_Pullback_v1
Hypothesis: Combine 6h ADX trend strength with Elder Ray (Bull/Bear Power) pullback entries in the direction of the 1d trend. 
In strong 1d uptrend (ADX>25 on 1d), wait for 6h Bull Power to turn positive after a dip (Bear Power<0 then >0) for longs. 
In strong 1d downtrend (ADX>25 on 1d), wait for 6h Bear Power to turn negative after a rally (Bull Power>0 then <0) for shorts. 
This captures momentum continuations after pullbacks in strong trends, working in both bull and bear markets by following the 1d ADX trend filter.
Uses discrete position sizing (0.25) to target ~20-40 trades/year.
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
    
    # Get 1d data for HTF trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilde_rma(values, period):
        """Wilder's RMA (same as EMA with alpha=1/period)"""
        return pd.Series(values).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    period = 14
    tr_mask = ~np.isnan(tr)
    if tr_mask.sum() < period:
        return np.zeros(n)
    atr = wilde_rma(tr, period)
    plus_dm_smooth = wilde_rma(plus_dm, period)
    minus_dm_smooth = wilde_rma(minus_dm, period)
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilde_rma(dx, period)
    
    # Align 1d ADX to 6h timeframe (need completed 1d bar + extra delay for ADX smoothing)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx, additional_delay_bars=2)
    
    # Calculate Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Previous values for crossover detection
    bull_power_prev = np.roll(bull_power, 1)
    bear_power_prev = np.roll(bear_power, 1)
    bull_power_prev[0] = np.nan
    bear_power_prev[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ADX and EMA
    start_idx = max(34, 13)  # ADX needs ~34 bars (14*2+6), EMA13 needs 13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(bull_power_prev[i]) or
            np.isnan(bear_power_prev[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend strength: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Look for pullback reversal signals in direction of strong trend
            # Long: Bear Power turns positive after being negative (bullish reversal)
            long_signal = strong_trend and (bear_power[i] > 0) and (bear_power_prev[i] <= 0)
            # Short: Bull Power turns negative after being positive (bearish reversal)
            short_signal = strong_trend and (bull_power[i] < 0) and (bull_power_prev[i] >= 0)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when Bear Power turns negative again or trend weakens
            exit_signal = (bear_power[i] < 0) or (adx_aligned[i] < 20)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when Bull Power turns positive again or trend weakens
            exit_signal = (bull_power[i] > 0) or (adx_aligned[i] < 20)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Trend_ElderRay_Pullback_v1"
timeframe = "6h"
leverage = 1.0