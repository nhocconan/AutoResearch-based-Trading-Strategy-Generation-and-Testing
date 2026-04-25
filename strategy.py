#!/usr/bin/env python3
"""
12h Camarilla Pivot H3L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance. 
Breakouts above H3 or below L3 with volume confirmation and aligned 1d EMA34 trend 
capture momentum moves. Chop filter (EWMA-based) avoids range-bound false breakouts. 
Designed for 12h timeframe with 50-150 total trades over 4 years to balance 
opportunity and fee drag. Works in bull/bear via trend filter and volume confirmation.
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
    
    # Get daily data for Camarilla pivots and EMA34 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient daily data
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_prev = high_prev - low_prev
    
    # Camarilla H3 and L3 levels
    H3 = close_prev + range_prev * 1.1 / 4.0
    L3 = close_prev - range_prev * 1.1 / 4.0
    
    # Align HTF levels to LTF (12h) with proper delay for completed bar
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate EWMA-based chop filter (12h) - avoids ranging markets
    # High-Low ratio relative to EWMA of ATR
    atr_raw = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    atr_raw[0] = high[0] - low[0]
    atr_ewma = pd.Series(atr_raw).ewm(span=14, adjust=False).values
    hl_ratio = (high - low) / atr_ewma
    # Chop filter: avoid when HL ratio is too low (ranging market)
    chop_ma = pd.Series(hl_ratio).ewm(span=20, adjust=False).mean().values
    chop_filter = chop_ma > 0.8  # Only trade when volatility is expanding
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, volume MA, and pivot calculation
    start_idx = max(35, 20)  # 35 for EMA34 (34+1), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        H3_val = H3_aligned[i]
        L3_val = L3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        chop_ok = chop_filter[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above H3 with volume confirmation in uptrend and chop filter OK
            long_breakout = (curr_close > H3_val) and volume_confirm and uptrend and chop_ok
            # Short: price breaks below L3 with volume confirmation in downtrend and chop filter OK
            short_breakout = (curr_close < L3_val) and volume_confirm and downtrend and chop_ok
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below H3 OR EMA34 trend turns down OR chop filter fails
            if curr_close < H3_val or curr_close < ema_34_val or not chop_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above L3 OR EMA34 trend turns up OR chop filter fails
            if curr_close > L3_val or curr_close > ema_34_val or not chop_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0