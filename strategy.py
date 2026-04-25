#!/usr/bin/env python3
"""
12h Camarilla Pivot R1/S1 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R1/S1) act as intraday support/resistance on 12h chart.
Breakout above R1 with 1d uptrend (EMA34) and volume spike signals bullish momentum.
Breakdown below S1 with 1d downtrend and volume spike signals bearish momentum.
Uses 12h timeframe with 1d HTF for trend filter. Targets 50-150 total trades over 4 years (12-37/year).
Works in both bull and bear markets: trend filter ensures we only trade with higher timeframe momentum,
while volume confirmation avoids false breakouts. Discrete position sizing (0.25) minimizes fee churn.
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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (based on previous 1d candle)
    # Pivot = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1 / 12
    # S1 = Pivot - (H - L) * 1.1 / 12
    df_1d = df_1d.copy()
    df_1d['pivot'] = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    df_1d['r1'] = df_1d['pivot'] + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    df_1d['s1'] = df_1d['pivot'] - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    
    pivot_1d = df_1d['pivot'].values
    r1_1d = df_1d['r1'].values
    s1_1d = df_1d['s1'].values
    
    # Align 1d levels to 12h timeframe (previous day's levels available after 1d close)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 24-period volume MA for 12h volume confirmation (24 periods = 12 days of 12h data)
    vol_ma_24_12h = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24_12h[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_ma_24_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        pivot_val = pivot_1d_aligned[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        vol_ma_12h = vol_ma_24_12h[i]
        
        # Volume confirmation: current 12h volume > 2.0 * 24-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: Break above R1 AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_high > r1_val and 
                         curr_close > ema_trend and volume_confirm)
            # Short: Break below S1 AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_low < s1_val and 
                          curr_close < ema_trend and volume_confirm)
            
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

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0