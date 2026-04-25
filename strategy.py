#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on daily timeframe with 1-week EMA50 trend filter and volume confirmation. 
In bull markets: buy when price breaks above Camarilla R1 and price > weekly EMA50. 
In bear markets: sell when price breaks below Camarilla S1 and price < weekly EMA50. 
Requires volume > 1.5x 20-day average for confirmation. 
Exit on opposite Camarilla level touch or trend reversal. 
Position size: 0.25 to limit drawdown. 
Target: 15-25 trades/year to stay within 1d hard max of 150 total trades.
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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume average
        return np.zeros(n)
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate weekly EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-day average volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Camarilla levels for previous 1d bar (use i-1 to avoid look-ahead)
        prev_idx = i - 1
        if prev_idx < 0:
            continue
            
        # Get previous day's OHLC from 1d data
        # Find the index in df_1d that corresponds to prices index i-1
        # Since we're using daily timeframe, we need to map 1d index to 1d bar
        # For daily timeframe, each price bar is 1 day, so we can use direct indexing with offset
        # But we need to be careful about alignment
        
        # Simpler approach: calculate Camarilla from 1d data and align
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate typical Camarilla levels (R1, S1) based on previous day
        # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
        # But we need to use the previous completed day's data
        
        # For simplicity in daily timeframe, we'll use rolling window on 1d data
        # and then align to match our prices index
        
        # Calculate Camarilla levels for each 1d bar
        hl_range_1d = high_1d - low_1d
        r1_1d = close_1d + (1.1 * hl_range_1d / 12)
        s1_1d = close_1d - (1.1 * hl_range_1d / 12)
        
        # Align Camarilla levels to match prices index
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
        
        # Determine 1w HTF trend (bullish = price above weekly EMA50)
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 1w uptrend + volume confirmation
            long_setup = (close[i] > r1_aligned[i]) and htf_1w_bullish and volume_confirm
            
            # Short setup: price breaks below Camarilla S1 + 1w downtrend + volume confirmation
            short_setup = (close[i] < s1_aligned[i]) and htf_1w_bearish and volume_confirm
            
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
            # Exit: price touches Camarilla S1 (stop) OR 1w trend turns bearish
            if (close[i] <= s1_aligned[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 (stop) OR 1w trend turns bullish
            if (close[i] >= r1_aligned[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0