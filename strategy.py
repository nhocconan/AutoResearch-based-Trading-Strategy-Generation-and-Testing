#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1w EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla pivot levels (H3/L3) act as institutional support/resistance. 
Breakouts above H3 or below L3 with 1w EMA50 trend alignment capture strong momentum. 
Volume spike confirms institutional participation. Chop filter (CHOP > 61.8) avoids ranging markets. 
Works in bull markets via buying H3 breakouts, bear markets via selling L3 breakdowns. 
Discrete position sizing (0.25) controls drawdown. Target: 19-50 trades/year on 4h.
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
    
    # Get 1d data for Camarilla pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_h3 = np.zeros(n)
    camarilla_l3 = np.zeros(n)
    for i in range(n):
        if i == 0:
            camarilla_h3[i] = 0.0
            camarilla_l3[i] = 0.0
        else:
            # Use previous 1d bar's OHLC (aligned to current 4h bar)
            idx_1d = i // 96  # Approximate: 96 * 15m = 24h, but we use HTF alignment properly below
            # Instead, we'll compute daily and align properly
            camarilla_h3[i] = 0.0
            camarilla_l3[i] = 0.0
    
    # Proper MTF: compute Camarilla on 1d data then align
    if len(df_1d) >= 1:
        # Calculate Camarilla levels for each 1d bar
        camarilla_h3_1d = np.zeros(len(df_1d))
        camarilla_l3_1d = np.zeros(len(df_1d))
        for j in range(len(df_1d)):
            if j == 0:
                camarilla_h3_1d[j] = 0.0
                camarilla_l3_1d[j] = 0.0
            else:
                high_prev = df_1d['high'].iloc[j-1]
                low_prev = df_1d['low'].iloc[j-1]
                close_prev = df_1d['close'].iloc[j-1]
                range_prev = high_prev - low_prev
                camarilla_h3_1d[j] = close_prev + range_prev * 1.1 / 4
                camarilla_l3_1d[j] = close_prev - range_prev * 1.1 / 4
        
        # Align to 4h timeframe
        camarilla_h3 = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
        camarilla_l3 = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = np.abs(np.diff(high, prepend=high[0]))
        tr2 = np.abs(np.diff(close, prepend=close[0]))
        tr3 = np.abs(np.diff(low, prepend=low[0]))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.zeros(n)
        for i in range(n):
            if i < 13:
                atr[i] = 0.0
            else:
                atr[i] = np.mean(tr[max(0, i-13):i+1])
    else:
        atr = np.zeros(n)
    
    # Pre-compute 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    # Calculate Choppiness Index (14) for regime filter
    chop = np.full(n, 50.0)  # default to neutral
    if n >= 14:
        # Calculate True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(np.subtract(high, np.concatenate([[close[0]], close[:-1]])))
        tr3 = np.abs(np.subtract(low, np.concatenate([[close[0]], close[:-1]])))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Calculate ATR sum for CHOP denominator
        atr_sum = np.zeros(n)
        for i in range(n):
            if i < 13:
                atr_sum[i] = 0.0
            else:
                atr_sum[i] = np.sum(tr[max(0, i-13):i+1])
        
        # Calculate highest high and lowest low over 14 periods
        hh = np.zeros(n)
        ll = np.zeros(n)
        for i in range(n):
            start_idx = max(0, i - 13)
            hh[i] = np.max(high[start_idx:i+1])
            ll[i] = np.min(low[start_idx:i+1])
        
        # Calculate Choppiness Index
        for i in range(n):
            if i >= 13 and atr_sum[i] > 0 and hh[i] > ll[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for indicators to propagate
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(chop[i]) or
            camarilla_h3[i] == 0.0 or camarilla_l3[i] == 0.0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50 = ema_50_1w_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        # Chop filter: avoid extreme chop (CHOP > 61.8) - ranging markets
        chop_filter = chop_val < 61.8
        
        if position == 0:
            # Long: break above Camarilla H3 AND uptrend AND volume spike AND chop filter
            long_condition = curr_close > camarilla_h3[i] and curr_close > ema_50 and volume_spike and chop_filter
            # Short: break below Camarilla L3 AND downtrend AND volume spike AND chop filter
            short_condition = curr_close < camarilla_l3[i] and curr_close < ema_50 and volume_spike and chop_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price falls below EMA50 or chop becomes extreme
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < ema_50 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price rises above EMA50 or chop becomes extreme
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > ema_50 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0