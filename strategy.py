#!/usr/bin/env python3
"""
12h Camarilla R3/S3 Breakout with 1w EMA50 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) from 1w act as strong support/resistance on 12h.
Breakouts above R3 or below S3 with volume confirmation and 1w EMA50 trend filter
capture institutional order flow. Works in bull markets via long breakouts and
in bear markets via short breakdowns. Uses ATR-based trailing stop for risk control.
Target: 75-200 total trades over 4 years (19-50/year) on 12h timeframe.
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
    
    # Get 1w data for Camarilla pivots and EMA50 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels for 1w (based on previous week's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low),
    # R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low),
    # S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low),
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3 and S3 as primary breakout levels (stronger levels)
    cam_r3 = np.full(len(df_1w), np.nan)
    cam_s3 = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        if i == 0:
            continue  # Need previous week
        prev_high = df_1w['high'].iloc[i-1]
        prev_low = df_1w['low'].iloc[i-1]
        prev_close = df_1w['close'].iloc[i-1]
        rang = prev_high - prev_low
        if rang <= 0:
            continue
        cam_r3[i] = prev_close + 1.1 * rang
        cam_s3[i] = prev_close - 1.1 * rang
    
    # Align Camarilla levels to 12h timeframe
    cam_r3_aligned = align_htf_to_ltf(prices, df_1w, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1w, cam_s3)
    
    # Calculate 20-period volume MA for volume confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (12h)
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
    
    # Start index: need enough for EMA50_1w, Camarilla, volume MA, ATR to propagate
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(cam_r3_aligned[i]) or 
            np.isnan(cam_s3_aligned[i]) or 
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
        ema50_1w = ema_50_1w_aligned[i]
        cam_r3 = cam_r3_aligned[i]
        cam_s3 = cam_s3_aligned[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average (strict filter)
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long breakout: close above R3 with volume confirmation and 1w EMA50 uptrend
            long_breakout = (curr_close > cam_r3) and volume_confirm and (curr_close > ema50_1w)
            # Short breakdown: close below S3 with volume confirmation and 1w EMA50 downtrend
            short_breakout = (curr_close < cam_s3) and volume_confirm and (curr_close < ema50_1w)
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = curr_close - 1.5 * atr  # Initial stop
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = curr_close + 1.5 * atr  # Initial stop
        elif position == 1:
            # Update trailing stop: raise stop to highest high - 1.5*ATR
            atr_stop = max(atr_stop, curr_high - 1.5 * atr)
            # Exit long: price closes below trailing stop
            if curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop: lower stop to lowest low + 1.5*ATR
            atr_stop = min(atr_stop, curr_low + 1.5 * atr)
            # Exit short: price closes above trailing stop
            if curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0