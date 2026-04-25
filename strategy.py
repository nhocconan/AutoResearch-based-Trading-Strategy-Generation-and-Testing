#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_VolumeConfirm_v2
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 12h EMA34 trend filter and volume spike confirmation.
In bull markets: long when price breaks above R1 (resistance) with volume > 1.5x 20-bar average and price > 12h EMA34.
In bear markets: short when price breaks below S1 (support) with volume > 1.5x 20-bar average and price < 12h EMA34.
Exit on opposite Camarilla level touch (R1 for shorts, S1 for longs) or trend reversal.
Position size: 0.25 to balance risk and reward.
Target: 20-40 trades/year to stay well under 400-trade 4h hard max.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for HTF trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 1d data for Camarilla pivot calculation (standard practice)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (no extra delay needed for pivot levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 1.5x 20-bar average
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_sma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume SMA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend (bullish = price above EMA34)
        htf_12h_bullish = close[i] > ema_34_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above R1 + volume spike + 12h uptrend
            long_setup = (close[i] > camarilla_r1_aligned[i]) and volume_spike[i] and htf_12h_bullish
            
            # Short setup: price breaks below S1 + volume spike + 12h downtrend
            short_setup = (close[i] < camarilla_s1_aligned[i]) and volume_spike[i] and htf_12h_bearish
            
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
            # Exit: price touches S1 (stop) OR 12h trend turns bearish
            if (close[i] <= camarilla_s1_aligned[i]) or (not htf_12h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches R1 (stop) OR 12h trend turns bullish
            if (close[i] >= camarilla_r1_aligned[i]) or (htf_12h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0