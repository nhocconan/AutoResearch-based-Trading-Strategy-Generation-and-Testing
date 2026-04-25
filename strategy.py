#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike
Hypothesis: Trade 4h Camarilla R1/S1 breakouts in the direction of the daily EMA34 trend, with volume confirmation.
Only long when price breaks above Camarilla R1 AND daily close > daily EMA34 AND volume > 1.5 * ATR4h.
Only short when price breaks below Camarilla S1 AND daily close < daily EMA34 AND volume > 1.5 * ATR4h.
Uses discrete sizing 0.25 to limit fee drag. Target: 20-50 trades/year.
Daily EMA34 trend filter provides structural edge in both bull and bear markets by aligning with intermediate-term institutional trend.
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
    
    # Get daily data for Camarilla pivot levels and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla levels (R1/S1)
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla R1 = C + (H-L)*1.1/12
    # Camarilla S1 = C - (H-L)*1.1/12
    camarilla_r1_1d = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1_1d = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(c_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for volume confirmation (using 4h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 and ATR
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * ATR
        volume_confirm = volume[i] > 1.5 * atr[i]
        
        # Determine daily trend from EMA34
        # Bullish trend: daily close > EMA34
        # Bearish trend: daily close < EMA34
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, c_1d)[i]
        if np.isnan(daily_close_aligned):
            signals[i] = 0.0
            continue
            
        if daily_close_aligned > ema_34_aligned[i]:
            daily_trend = 'bullish'  # only allow longs
        elif daily_close_aligned < ema_34_aligned[i]:
            daily_trend = 'bearish'  # only allow shorts
        else:
            daily_trend = 'neutral'  # no trades in neutral zone
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 AND volume confirm AND bullish daily trend
            long_setup = (close[i] > camarilla_r1_aligned[i]) and volume_confirm and (daily_trend == 'bullish')
            
            # Short setup: price breaks below Camarilla S1 AND volume confirm AND bearish daily trend
            short_setup = (close[i] < camarilla_s1_aligned[i]) and volume_confirm and (daily_trend == 'bearish')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Camarilla S1 OR daily trend turns bearish
            if (close[i] < camarilla_s1_aligned[i]) or (daily_trend == 'bearish'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Camarilla R1 OR daily trend turns bullish
            if (close[i] > camarilla_r1_aligned[i]) or (daily_trend == 'bullish'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0