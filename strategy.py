#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 level breakout with 1d EMA34 trend filter and volume spike.
Camarilla pivot levels provide institutional support/resistance. Breakout above R1 or below S1
with volume confirmation and aligned daily trend captures momentum. Uses 1d EMA34 for trend
filter to avoid counter-trend trades. Designed for ~30-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla levels, EMA34, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1_level = close_1d + camarilla_range
    s1_level = close_1d - camarilla_range
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: volume / 20-period average volume
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20
    
    # Align all indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.5  # Volume must be 1.5x average
        
        if position == 0:
            # Enter long: price breaks above R1, volume spike, uptrend
            if (price_close > r1 and 
                vol_ratio > vol_threshold and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, volume spike, downtrend
            elif (price_close < s1 and 
                  vol_ratio > vol_threshold and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to previous day's close or trend reversal
            prev_close = close_1d_aligned[i] if 'close_1d_aligned' in locals() else None
            if prev_close is None:
                # Align previous day's close
                close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
                prev_close = close_1d_aligned[i]
            
            if position == 1 and (price_close < prev_close or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > prev_close or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0