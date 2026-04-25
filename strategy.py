#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h timeframe with 1-day EMA34 trend filter, volume confirmation, and choppiness regime filter. 
In trending markets (CHOP < 38.2): buy when price breaks above Camarilla R1 and price > daily EMA34; sell when price breaks below Camarilla S1 and price < daily EMA34. 
In ranging markets (CHOP >= 38.2): fade the breakout - sell when price breaks above R1 and price > daily EMA34; buy when price breaks below S1 and price < daily EMA34. 
Requires volume > 1.3x 20-period average for confirmation. 
Exit on opposite Camarilla level touch or trend reversal. 
Position size: 0.25 to limit drawdown and reduce fee churn. 
Target: 50-150 total trades over 4 years = 12-37/year. 
Uses 12h primary timeframe with 1d HTF for multi-timeframe alignment.
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Calculate daily EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate daily choppiness index for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(sumTR14 / (ATR14 * 14)) / log10(14)
    chop_raw = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    chop_1d = chop_raw
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Camarilla levels (R1, S1) for each 1d bar
    hl_range_1d = high_1d - low_1d
    r1_1d = close_1d + (1.1 * hl_range_1d / 12)  # R1 = close + 1.1*(high-low)/12
    s1_1d = close_1d - (1.1 * hl_range_1d / 12)  # S1 = close - 1.1*(high-low)/12
    
    # Align Camarilla levels to match prices index
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), volume MA (20), and chop (14)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above daily EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        # Regime filter: CHOP < 38.2 = trending, CHOP >= 38.2 = ranging
        is_trending = chop_1d_aligned[i] < 38.2
        is_ranging = chop_1d_aligned[i] >= 38.2
        
        if position == 0:
            if is_trending:
                # Trending market: follow the breakout
                # Long setup: price breaks above Camarilla R1 + 1d uptrend + volume confirmation
                long_setup = (close[i] > r1_aligned[i]) and htf_1d_bullish and volume_confirm
                
                # Short setup: price breaks below Camarilla S1 + 1d downtrend + volume confirmation
                short_setup = (close[i] < s1_aligned[i]) and htf_1d_bearish and volume_confirm
            else:
                # Ranging market: fade the breakout (mean reversion)
                # Short setup: price breaks above Camarilla R1 + 1d uptrend + volume confirmation
                short_setup = (close[i] > r1_aligned[i]) and htf_1d_bullish and volume_confirm
                
                # Long setup: price breaks below Camarilla S1 + 1d downtrend + volume confirmation
                long_setup = (close[i] < s1_aligned[i]) and htf_1d_bearish and volume_confirm
            
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
            if (close[i] <= s1_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 (stop) OR 1d trend turns bullish
            if (close[i] >= r1_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0