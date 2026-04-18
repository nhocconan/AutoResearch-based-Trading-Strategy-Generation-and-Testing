#!/usr/bin/env python3
"""
4h Monthly Pivot R2/S2 Breakout with Volume Spike and 12h EMA Trend Filter
Hypothesis: Monthly pivot levels (R2, S2) act as strong monthly support/resistance.
Breakouts beyond these levels with volume confirmation and 12h EMA trend filter capture momentum.
Designed for 20-50 trades/year on 4h timeframe with low trade frequency to minimize fee drag.
Works in bull/bear markets by requiring volume spike and 12h EMA trend filter.
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
    
    # Get monthly data for pivot calculation (once before loop)
    df_m = get_htf_data(prices, '1M')
    
    # Calculate monthly pivot points using standard formula
    # P = (H + L + C) / 3
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # Using previous month's data to avoid look-ahead
    monthly_high = df_m['high']
    monthly_low = df_m['low']
    monthly_close = df_m['close']
    
    pivot = (monthly_high + monthly_low + monthly_close) / 3
    r2 = pivot + (monthly_high - monthly_low)
    s2 = pivot - (monthly_high - monthly_low)
    
    # Shift by 1 to use previous month's levels only
    r2_prev = r2.shift(1).values
    s2_prev = s2.shift(1).values
    
    # Align to 4h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_m, r2_prev)
    s2_aligned = align_htf_to_ltf(prices, df_m, s2_prev)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike: 2x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        ema_trend = ema_34_aligned[i]
        
        if position == 0:
            # Long: break above R2 with volume spike and price above 12h EMA (uptrend)
            if price > r2_val and volume_spike[i] and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S2 with volume spike and price below 12h EMA (downtrend)
            elif price < s2_val and volume_spike[i] and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to S2 or breaks below 12h EMA
            if price <= s2_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to R2 or breaks above 12h EMA
            if price >= r2_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_MonthlyPivot_R2S2_Breakout_Volume_12hEMA"
timeframe = "4h"
leverage = 1.0