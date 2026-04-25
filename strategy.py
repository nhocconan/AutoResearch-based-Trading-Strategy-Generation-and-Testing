#!/usr/bin/env python3
"""
1d Camarilla Pivot H3/L3 Breakout with 1w EMA50 Trend and Volume Spike
Hypothesis: Weekly Camarilla H3/L3 levels act as strong support/resistance on 1d.
Breakouts above H3 or below L3 with volume confirmation and 1w EMA50 trend filter
capture institutional order flow. Works in bull markets via long breakouts and
in bear markets via short breakdowns. Uses ATR-based trailing stop for risk control.
Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
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
    
    # Get 1w data for Camarilla pivots and EMA50 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels for 1w (based on previous week's OHLC)
    # Camarilla: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    # We use H3 and L3 as primary breakout levels
    camarilla_h3 = np.full(len(df_1w), np.nan)
    camarilla_l3 = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        if i == 0:
            continue  # Need previous week
        prev_high = df_1w['high'].iloc[i-1]
        prev_low = df_1w['low'].iloc[i-1]
        prev_close = df_1w['close'].iloc[i-1]
        rang = prev_high - prev_low
        if rang <= 0:
            continue
        camarilla_h3[i] = prev_close + 1.1 * rang
        camarilla_l3[i] = prev_close - 1.1 * rang
    
    # Align Camarilla levels to 1d timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Calculate 20-period volume MA for volume confirmation (1d)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (1d)
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
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
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
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average (strict filter)
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long breakout: close above H3 with volume confirmation and 1w EMA50 uptrend
            long_breakout = (curr_close > h3) and volume_confirm and (curr_close > ema50_1w)
            # Short breakdown: close below L3 with volume confirmation and 1w EMA50 downtrend
            short_breakout = (curr_close < l3) and volume_confirm and (curr_close < ema50_1w)
            
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

name = "1d_Camarilla_H3_L3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0