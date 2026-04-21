#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: Price breaking above/below Camarilla R1/S1 levels with 1d EMA34 trend filter and volume spike. Captures institutional breakouts in trending markets while avoiding chop. Works in bull/bear by following 1d trend direction. Target 25-40 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Camarilla levels from previous 1d bar ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r1 = np.zeros(len(df_1d))
    camarilla_s1 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_r1[i] = 0
            camarilla_s1[i] = 0
        else:
            # Camarilla formulas using previous day's OHLC
            camarilla_r1[i] = close_1d_prev[i-1] + 1.1 * (high_1d[i-1] - low_1d[i-1]) / 12
            camarilla_s1[i] = close_1d_prev[i-1] - 1.1 * (high_1d[i-1] - low_1d[i-1]) / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        trend_1d = ema_34_1d_aligned[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Price breaks above R1 + volume spike + price above 1d EMA34
            if (price_high > camarilla_r1_val and 
                vol_spike > 1.5 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume spike + price below 1d EMA34
            elif (price_low < camarilla_s1_val and 
                  vol_spike > 1.5 and 
                  price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to Camarilla H4/L4 levels (mean reversion)
            # Calculate H4 and L4 for exit
            camarilla_h4 = camarilla_r1_val + 1.1 * (high_1d[i-1] - low_1d[i-1]) / 6 if i > 0 else camarilla_r1_val
            camarilla_l4 = camarilla_s1_val - 1.1 * (high_1d[i-1] - low_1d[i-1]) / 6 if i > 0 else camarilla_s1_val
            
            if position == 1 and price_low < camarilla_h4:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_high > camarilla_l4:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0