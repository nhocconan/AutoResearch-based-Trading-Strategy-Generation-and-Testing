#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v2
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h timeframe with 1-day EMA34 trend filter and volume confirmation.
In bull markets: buy when price breaks above Camarilla R1 and price > daily EMA34.
In bear markets: sell when price breaks below Camarilla S1 and price < daily EMA34.
Requires volume > 1.5x 20-bar average for confirmation.
Exit on opposite Camarilla level touch or trend reversal.
Position size: 0.25 to limit drawdown.
Target: 30-50 trades/year to stay within 4h hard max of 200 total trades.
Uses proven Camarilla breakout structure with volume confirmation and trend filter.
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
    
    # Get 1d data for HTF trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34 and volume MA
        return np.zeros(n)
    
    # Calculate daily EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-bar average volume for confirmation (on 1d data)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 4h Camarilla levels (R1, S1) from previous 4h bar
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    hl_range_4h = high_4h - low_4h
    r1_4h = close_4h + (1.1 * hl_range_4h / 12)
    s1_4h = close_4h - (1.1 * hl_range_4h / 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above daily EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-bar average
        volume_confirm = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 1d uptrend + volume confirmation
            long_setup = (close[i] > r1_4h[i]) and htf_1d_bullish and volume_confirm
            
            # Short setup: price breaks below Camarilla S1 + 1d downtrend + volume confirmation
            short_setup = (close[i] < s1_4h[i]) and htf_1d_bearish and volume_confirm
            
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
            # Exit: price touches Camarilla S1 (stop) OR 1d trend turns bearish
            if (close[i] <= s1_4h[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 (stop) OR 1d trend turns bullish
            if (close[i] >= r1_4h[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0