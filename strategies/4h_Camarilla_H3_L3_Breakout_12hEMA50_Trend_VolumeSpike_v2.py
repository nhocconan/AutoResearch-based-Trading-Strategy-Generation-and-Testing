#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout with 12h EMA50 Trend and Volume Spike Confirmation
Hypothesis: Camarilla pivot levels (H3/L3) from 1d act as strong support/resistance on 4h.
Breakouts above H3 or below L3 with volume confirmation (>2x 20-bar vol MA) and 12h EMA50 trend
filter capture strong momentum moves. Uses ATR-based trailing stop (2.0*ATR) for risk control.
Tight entry conditions target 75-200 total trades over 4 years to avoid fee drag. Works in
bull markets via long breakouts and in bear markets via short breakdowns. EMA50 on 12h provides
smoother trend filter than shorter EMAs, reducing whipsaws in choppy markets and improving
generalization to bear markets (2025+ test period).
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
    
    # Get 1d data for Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d (based on previous day's OHLC)
    # Camarilla: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    cam_h3 = np.full(len(df_1d), np.nan)
    cam_l3 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        prev_high = df_1d['high'].iloc[i-1]
        prev_low = df_1d['low'].iloc[i-1]
        prev_close = df_1d['close'].iloc[i-1]
        rang = prev_high - prev_low
        if rang <= 0:
            continue
        cam_h3[i] = prev_close + 1.1 * rang
        cam_l3[i] = prev_close - 1.1 * rang
    
    # Align Camarilla levels to 4h timeframe
    cam_h3_aligned = align_htf_to_ltf(prices, df_1d, cam_h3)
    cam_l3_aligned = align_htf_to_ltf(prices, df_1d, cam_l3)
    
    # Get 12h data for EMA50 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period volume MA for volume confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (4h)
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for Camarilla, EMA50_12h, volume MA, ATR to propagate
    start_idx = max(2, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cam_h3_aligned[i]) or 
            np.isnan(cam_l3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        cam_h3 = cam_h3_aligned[i]
        cam_l3 = cam_l3_aligned[i]
        ema50_12h = ema_50_12h_aligned[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average (strict filter)
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long breakout: close above H3 with volume confirmation and 12h EMA50 uptrend
            long_breakout = (curr_close > cam_h3) and volume_confirm and (curr_close > ema50_12h)
            # Short breakdown: close below L3 with volume confirmation and 12h EMA50 downtrend
            short_breakout = (curr_close < cam_l3) and volume_confirm and (curr_close < ema50_12h)
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = curr_close - 2.0 * atr  # Initial stop
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = curr_close + 2.0 * atr  # Initial stop
        elif position == 1:
            # Update trailing stop: raise stop to highest high - 2.0*ATR
            atr_stop = max(atr_stop, curr_high - 2.0 * atr)
            # Exit long: price closes below trailing stop
            if curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop: lower stop to lowest low + 2.0*ATR
            atr_stop = min(atr_stop, curr_low + 2.0 * atr)
            # Exit short: price closes above trailing stop
            if curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3_L3_Breakout_12hEMA50_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0