#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3L3 levels on 4h act as strong support/resistance. Breakouts above H3 or below L3 with volume confirmation (>2x 20-period volume MA) and low chop (<61.8) capture momentum in trending markets. 1d EMA34 filter ensures alignment with daily trend to avoid counter-trend trades. Designed for 4h timeframe targeting 75-200 total trades over 4 years. Works in both bull and bear markets via daily trend filter and chop regime filter to reduce false breakouts in sideways markets.
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
    
    # Get 1d data for EMA34 trend and chop filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 days for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d chop filter (Choppiness Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    atr_1d = np.zeros(len(df_1d))
    tr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d_arr[i-1]), abs(low_1d[i] - close_1d_arr[i-1]))
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    max_hh = np.zeros(len(df_1d))
    min_ll = np.zeros(len(df_1d))
    for i in range(14, len(df_1d)):
        max_hh[i] = np.max(high_1d[i-13:i+1])
        min_ll[i] = np.min(low_1d[i-13:i+1])
    
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if max_hh[i] != min_ll[i]:
            sum_atr = np.sum(atr_1d[i-13:i+1])
            chop_1d[i] = 100 * np.log10(sum_atr / (max_hh[i] - min_ll[i])) / np.log10(14)
        else:
            chop_1d[i] = 0
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (4h)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate Camarilla levels for 4h (using previous 4h bar's high, low, close)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    for i in range(1, n):
        # Use previous bar's HLC to calculate today's levels (no look-ahead)
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        rang = phigh - plow
        camarilla_h3[i] = pclose + rang * 1.1 / 4
        camarilla_l3[i] = pclose - rang * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34, volume MA, ATR, and Camarilla
    start_idx = max(34, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        camarilla_h3_val = camarilla_h3[i]
        camarilla_l3_val = camarilla_l3[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Chop filter: low chop indicates trending market (chop < 61.8)
        trending_market = chop_val < 61.8
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals at Camarilla H3/L3 levels
            # Long: price breaks above Camarilla H3 with volume confirmation in uptrend and trending market
            long_breakout = (curr_close > camarilla_h3_val) and volume_confirm and uptrend and trending_market
            # Short: price breaks below Camarilla L3 with volume confirmation in downtrend and trending market
            short_breakout = (curr_close < camarilla_l3_val) and volume_confirm and downtrend and trending_market
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * atr_val
            # Exit conditions: price closes below Camarilla L3 OR stoploss hit OR EMA34 trend turns down OR chop becomes too high (range)
            if curr_close < camarilla_l3_val or curr_close < stop_loss or curr_close < ema_34_val or chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * atr_val
            # Exit conditions: price closes above Camarilla H3 OR stoploss hit OR EMA34 trend turns up OR chop becomes too high (range)
            if curr_close > camarilla_h3_val or curr_close > stop_loss or curr_close > ema_34_val or chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0