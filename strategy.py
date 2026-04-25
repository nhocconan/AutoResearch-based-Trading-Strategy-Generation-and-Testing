#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeConfirm_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 1h timeframe with 4h EMA34 trend filter and volume confirmation. 
In bull markets: buy when price breaks above Camarilla R1 and price > 4h EMA34. 
In bear markets: sell when price breaks below Camarilla S1 and price < 4h EMA34. 
Requires volume > 1.3x 20-period average for confirmation. 
Exit on opposite Camarilla level touch or trend reversal. 
Position size: 0.20 to limit drawdown and reduce fee churn. 
Target: 60-150 total trades over 4 years = 15-37/year. 
Uses 4h for signal direction, 1h only for entry timing precision. 
Session filter: 08-20 UTC to reduce noise trades.
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
    
    # Get 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need sufficient data for volume average
        return np.zeros(n)
    
    # Calculate 4h EMA34 for HTF trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 20-period average volume for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # Calculate Camarilla levels for each 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    hl_range_4h = high_4h - low_4h
    r1_4h = close_4h + (1.1 * hl_range_4h / 12)  # R1 = close + 1.1*(high-low)/12
    s1_4h = close_4h - (1.1 * hl_range_4h / 12)  # S1 = close - 1.1*(high-low)/12
    
    # Align Camarilla levels to match prices index
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h HTF trend (bullish = price above 4h EMA34)
        htf_4h_bullish = close[i] > ema_34_4h_aligned[i]
        htf_4h_bearish = close[i] < ema_34_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 4h uptrend + volume confirmation
            long_setup = (close[i] > r1_aligned[i]) and htf_4h_bullish and volume_confirm
            
            # Short setup: price breaks below Camarilla S1 + 4h downtrend + volume confirmation
            short_setup = (close[i] < s1_aligned[i]) and htf_4h_bearish and volume_confirm
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price touches Camarilla S1 (stop) OR 4h trend turns bearish
            if (close[i] <= s1_aligned[i]) or (not htf_4h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price touches Camarilla R1 (stop) OR 4h trend turns bullish
            if (close[i] >= r1_aligned[i]) or (htf_4h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0