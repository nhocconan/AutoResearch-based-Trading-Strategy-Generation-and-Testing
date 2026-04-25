#!/usr/bin/env python3
"""
1d_Camarilla_H3L3_Breakout_1wTrendFilter_VolumeConfirm_v1
Hypothesis: Trade Camarilla H3/L3 breakouts on daily timeframe with 1-week EMA50 trend filter and volume confirmation.
Camarilla levels from weekly chart provide institutional support/resistance. Breakouts above H3 or below L3
with 1-week EMA trend alignment and volume spike capture momentum moves. Using 1d primary timeframe to reduce
trade frequency and fee drag, targeting 7-25 trades/year per symbol.
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
    
    # Get 1w data for HTF Camarilla pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1w bar (HLC of completed weekly candle)
    # Camarilla: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # where C = (H+L+C)/3 (typical price)
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    typical_price_1w = (h_1w + l_1w + c_1w) / 3.0
    range_1w = h_1w - l_1w
    camarilla_h3_1w = typical_price_1w + (range_1w * 1.1 / 4.0)
    camarilla_l3_1w = typical_price_1w - (range_1w * 1.1 / 4.0)
    
    # Align Camarilla levels to 1d timeframe (use previous week's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    # Volume confirmation: 1d volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above EMA50)
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla H3 + 1w uptrend + volume spike
            long_setup = (close[i] > camarilla_h3_aligned[i]) and htf_1w_bullish and volume_spike[i]
            
            # Short setup: price breaks below Camarilla L3 + 1w downtrend + volume spike
            short_setup = (close[i] < camarilla_l3_aligned[i]) and htf_1w_bearish and volume_spike[i]
            
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
            # Exit: price touches Camarilla L3 (stop) OR 1w trend turns bearish
            if (close[i] <= camarilla_l3_aligned[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla H3 (stop) OR 1w trend turns bullish
            if (close[i] >= camarilla_h3_aligned[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wTrendFilter_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0