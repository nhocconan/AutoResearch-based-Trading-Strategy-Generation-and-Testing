#!/usr/bin/env python3
"""
12h_WilliamsAlligator_1wTrend_Volume
Hypothesis: Williams Alligator on 12h timeframe with 1w trend filter and volume confirmation.
The Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
Price trading outside the Alligator's mouth indicates a trend; price inside indicates consolidation.
We only trade when price is outside the mouth in the direction of the 1w trend, with volume confirmation.
This filters out false breakouts in ranging markets. Works in both bull (buy signals during uptrends)
and bear (sell signals during downtrends). Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "12h_WilliamsAlligator_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams Alligator components (13,8,5 periods with future shifts)
    # Jaw: 13-period SMMA, shifted by 8 bars
    jaw_period = 13
    jaw_shift = 8
    jaw_raw = np.full(n, np.nan)
    if n >= jaw_period:
        jaw_raw[jaw_period-1] = np.mean(high[:jaw_period] + low[:jaw_period]) / 2
        for i in range(jaw_period, n):
            jaw_raw[i] = (jaw_raw[i-1] * (jaw_period-1) + (high[i] + low[i]) / 2) / jaw_period
    jaw = np.full(n, np.nan)
    if n >= jaw_period + jaw_shift:
        jaw[jaw_shift:] = jaw_raw[:-jaw_shift] if jaw_shift > 0 else jaw_raw
    
    # Teeth: 8-period SMMA, shifted by 5 bars
    teeth_period = 8
    teeth_shift = 5
    teeth_raw = np.full(n, np.nan)
    if n >= teeth_period:
        teeth_raw[teeth_period-1] = np.mean(high[:teeth_period] + low[:teeth_period]) / 2
        for i in range(teeth_period, n):
            teeth_raw[i] = (teeth_raw[i-1] * (teeth_period-1) + (high[i] + low[i]) / 2) / teeth_period
    teeth = np.full(n, np.nan)
    if n >= teeth_period + teeth_shift:
        teeth[teeth_shift:] = teeth_raw[:-teeth_shift] if teeth_shift > 0 else teeth_raw
    
    # Lips: 5-period SMMA, shifted by 3 bars
    lips_period = 5
    lips_shift = 3
    lips_raw = np.full(n, np.nan)
    if n >= lips_period:
        lips_raw[lips_period-1] = np.mean(high[:lips_period] + low[:lips_period]) / 2
        for i in range(lips_period, n):
            lips_raw[i] = (lips_raw[i-1] * (lips_period-1) + (high[i] + low[i]) / 2) / lips_period
    lips = np.full(n, np.nan)
    if n >= lips_period + lips_shift:
        lips[lips_shift:] = lips_raw[:-lips_shift] if lips_shift > 0 else lips_raw
    
    # Volume confirmation: current 12h volume > 1.5x average 1w volume (scaled to 12h)
    volume_1w = df_1w['volume'].values
    vol_sma20_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 20:
        vol_sma20_1w[19] = np.mean(volume_1w[:20])
        for i in range(20, len(volume_1w)):
            vol_sma20_1w[i] = (vol_sma20_1w[i-1] * 19 + volume_1w[i]) / 20
    vol_sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period + jaw_shift, teeth_period + teeth_shift, lips_period + lips_shift, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_sma20_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 1w volume (scaled to 12h)
        # Approximate 12h volume from 1w: 1w volume / (7*2) since 168h/12h = 14
        vol_12h_approx = vol_sma20_1w_aligned[i] / 14.0
        volume_confirm = volume[i] > 1.5 * vol_12h_approx
        
        if position == 0:
            # Long: Price above Alligator's mouth (Lips > Teeth > Jaw) with uptrend and volume confirmation
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price below Alligator's mouth (Jaw > Teeth > Lips) with downtrend and volume confirmation
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and close[i] < ema50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price re-enters Alligator's mouth (Lips <= Teeth or Teeth <= Jaw) or trend reversal
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters Alligator's mouth (Jaw <= Teeth or Teeth <= Lips) or trend reversal
            if jaw[i] <= teeth[i] or teeth[i] <= lips[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals