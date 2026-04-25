#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike Confirmation + Chop Filter
Hypothesis: Camarilla H3/L3 levels act as strong support/resistance derived from prior day's range.
Breakouts above H3 or below L3 with volume confirmation, aligned 1d EMA34 trend, and 
choppy market filter (Choppiness Index > 61.8) capture strong momentum moves while 
avoiding false breakouts in ranging markets. Designed for BTC/ETH with 75-200 total 
trades over 4 years to balance opportunity and fee drag. Works in bull markets (trend 
continuation) and bear markets (trend continuation down) by using 1d EMA34 as trend 
filter and chop filter to avoid whipsaws.
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
    
    # Get daily data for Camarilla pivot calculation and EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need at least 35 days for EMA34 and pivot calculation
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to get previous day's data (avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # First day has no previous day
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate Camarilla levels: H3/L3 for entry, H4/L4 for stop
    camarilla_h3_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l3_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_h4_1d = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l4_1d = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate Choppiness Index (4h) for regime filter
    chop = np.full(n, np.nan)
    for i in range(34, n):
        # True range
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of true range over 34 periods
        atr_sum = np.sum(tr[i-33:i+1])
        
        # Highest high and lowest low over 34 periods
        hh = np.max(high[i-33:i+1])
        ll = np.min(low[i-33:i+1])
        
        # Choppiness Index formula
        if atr_sum > 0 and hh > ll:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(34)
        else:
            chop[i] = 50.0  # neutral value
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, volume MA, Camarilla levels, and chop
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        camarilla_h3 = camarilla_h3_aligned[i]
        camarilla_l3 = camarilla_l3_aligned[i]
        camarilla_h4 = camarilla_h4_aligned[i]
        camarilla_l4 = camarilla_l4_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Chop filter: only trade in choppy markets (Choppiness Index > 61.8 = ranging)
        chop_filter = chop_val > 61.8
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above H3 with volume confirmation in uptrend AND choppy market
            long_breakout = (curr_close > camarilla_h3) and volume_confirm and uptrend and chop_filter
            # Short: price breaks below L3 with volume confirmation in downtrend AND choppy market
            short_breakout = (curr_close < camarilla_l3) and volume_confirm and downtrend and chop_filter
            
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
            # Exit long: price closes below L3 OR EMA34 trend turns down OR chop breaks down (trending market)
            if curr_close < camarilla_l3 or curr_close < ema_34_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above H3 OR EMA34 trend turns up OR chop breaks down (trending market)
            if curr_close > camarilla_h3 or curr_close > ema_34_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0