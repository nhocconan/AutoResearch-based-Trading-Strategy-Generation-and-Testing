#!/usr/bin/env python3
"""
4h Camarilla Pivot R1S1 Breakout with Volume Spike and Daily Trend Filter
Hypothesis: Camarilla pivot levels (R1/S1) act as strong intraday support/resistance.
Breakouts with volume confirmation and daily EMA34 trend filter capture momentum moves.
Designed for 20-50 trades/year on 4h timeframe. Works in both bull and bear markets
by filtering trades with the daily trend and requiring volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and EMA (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for the day
    # Using previous day's OHLC to calculate today's levels
    # Shift by 1 to avoid look-ahead: use previous day's data
    prev_high = df_d['high'].shift(1).values
    prev_low = df_d['low'].shift(1).values
    prev_close = df_d['close'].shift(1).values
    
    # Camarilla pivot calculations
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Resistance and Support levels
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    r2 = pivot + (range_val * 1.1 / 6)
    s2 = pivot - (range_val * 1.1 / 6)
    
    # Align pivots to 4h timeframe (no additional delay needed as pivots are known at day open)
    r1_aligned = align_htf_to_ltf(prices, df_d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_d, s2)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_d, ema_34)
    
    # Volume spike: 2x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (4h ATR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema = ema_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above R1 with volume spike and price above EMA34 (uptrend)
            if price > r1_level and volume_spike[i] and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and price below EMA34 (downtrend)
            elif price < s1_level and volume_spike[i] and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to S1 or ATR trailing stop
            if price <= s1_level or price < (high[i] - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to R1 or ATR trailing stop
            if price >= r1_level or price > (low[i] + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeSpike_EMA34"
timeframe = "4h"
leverage = 1.0