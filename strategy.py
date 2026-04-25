#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout with 1d EMA34 Trend and Volume Spike + Choppiness Filter
Hypothesis: Camarilla pivot levels (R3/S3) act as stronger support/resistance on 4h chart.
Breakout above R3 with 1d uptrend (EMA34) and volume spike signals bullish momentum.
Breakdown below S3 with 1d downtrend and volume spike signals bearish momentum.
Choppiness filter (CHOP > 61.8) ensures we only trade in ranging markets where mean reversion at pivots works.
Uses 4h timeframe with 1d HTF for trend filter. Targets 75-200 total trades over 4 years (19-50/year).
Works in both bull and bear markets: trend filter ensures we only trade with higher timeframe momentum,
while volume confirmation and chop filter avoid false breakouts. Discrete position sizing (0.25) minimizes fee churn.
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
    
    # Get 1d data for EMA34 trend and Camarilla levels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (based on previous 1d candle)
    df_1d = df_1d.copy()
    df_1d['pivot'] = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    df_1d['r3'] = df_1d['pivot'] + (df_1d['high'] - df_1d['low']) * 1.1 / 4  # R3 = Pivot + (H-L)*1.1/4
    df_1d['s3'] = df_1d['pivot'] - (df_1d['high'] - df_1d['low']) * 1.1 / 4  # S3 = Pivot - (H-L)*1.1/4
    
    pivot_1d = df_1d['pivot'].values
    r3_1d = df_1d['r3'].values
    s3_1d = df_1d['s3'].values
    
    # Align 1d levels to 4h timeframe (previous day's levels available after 1d close)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate 24-period volume MA for 4h volume confirmation (24 periods = 4 days of 4h data)
    vol_ma_24_4h = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24_4h[i] = np.mean(volume[i-23:i+1])
    
    # Calculate Choppiness Index on 4h data (using 14-period)
    chop = np.full(n, np.nan)
    for i in range(14, n):
        # True Range
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr = max(tr1, tr2, tr3)
        
        # Sum of TR over 14 periods
        sum_tr = 0
        for j in range(14):
            idx = i - j
            tr1_j = high[idx] - low[idx]
            tr2_j = abs(high[idx] - close[idx-1]) if idx > 0 else 0
            tr3_j = abs(low[idx] - close[idx-1]) if idx > 0 else 0
            tr_j = max(tr1_j, tr2_j, tr3_j)
            sum_tr += tr_j
        
        # Highest high and lowest low over 14 periods
        hh = high[i-13:i+1].max() if i >= 13 else high[:i+1].max()
        ll = low[i-13:i+1].min() if i >= 13 else low[:i+1].min()
        
        if sum_tr > 0 and hh > ll:
            chop[i] = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, volume MA, and chop
    start_idx = max(34, 24, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(vol_ma_24_4h[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        pivot_val = pivot_1d_aligned[i]
        r3_val = r3_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        vol_ma_4h = vol_ma_24_4h[i]
        chop_val = chop[i]
        
        # Volume confirmation: current 4h volume > 2.0 * 24-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_4h
        # Choppiness filter: only trade when CHOP > 61.8 (ranging market)
        chop_filter = chop_val > 61.8
        
        if position == 0:
            # Look for entry signals
            # Long: Break above R3 AND price > EMA34 (uptrend) AND volume confirmation AND chop filter
            long_entry = (curr_high > r3_val and 
                         curr_close > ema_trend and volume_confirm and chop_filter)
            # Short: Break below S3 AND price < EMA34 (downtrend) AND volume confirmation AND chop filter
            short_entry = (curr_low < s3_val and 
                          curr_close < ema_trend and volume_confirm and chop_filter)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Price crosses below pivot OR EMA34 trend turns down
            if (curr_close < pivot_val or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Price crosses above pivot OR EMA34 trend turns up
            if (curr_close > pivot_val or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0