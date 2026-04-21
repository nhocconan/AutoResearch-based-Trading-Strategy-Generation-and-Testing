#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using weekly pivot points (R2/S2) with 1w EMA34 trend filter and volume confirmation.
In uptrend (price > EMA34), buy breakouts above weekly R2; in downtrend (price < EMA34), sell breakdowns below weekly S2.
Weekly R2/S2 provide stronger institutional support/resistance than R1/S1, reducing false breakouts.
EMA34 filters for stronger trend alignment; volume confirms breakout strength.
Works in bull markets (buy R2 breaks) and bear markets (sell S2 breaks). Target: 7-25 trades/year for 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's H/L/C)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R2 = Pivot + (High - Low)
    r2_1w = pivot_1w + (high_1w - low_1w)
    # S2 = Pivot - (High - Low)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly R2/S2 to 1d timeframe (wait for weekly bar to close)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Load weekly data ONCE before loop for EMA trend filter
    # Weekly EMA34 for trend filter (slower, more reliable)
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # 1d volume confirmation (volume spike > 1.8x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.8  # Higher volume spike filter for quality
        
        if position == 0:
            # Enter long: price breaks above weekly R2 + uptrend (price > EMA34) + volume spike
            if (price_close > r2_aligned[i] and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S2 + downtrend (price < EMA34) + volume spike
            elif (price_close < s2_aligned[i] and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (price crosses EMA34 in opposite direction)
            if position == 1 and price_close < ema_trend:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyPivot_R2S2_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0