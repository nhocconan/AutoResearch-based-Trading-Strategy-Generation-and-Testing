#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeRegime_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h timeframe with 1-day EMA34 trend filter, volume spike confirmation, and choppiness regime filter. 
In trending markets (CHOP < 38.2): buy when price breaks above Camarilla R1 and price > daily EMA34; sell when price breaks below Camarilla S1 and price < daily EMA34. 
In choppy markets (CHOP > 61.8): fade the breakout (short R1 break, long S1 break) with same filters. 
Requires volume > 2.0x 20-period average for confirmation. 
Position size: 0.25. 
Target: 80-180 total trades over 4 years = 20-45/year. 
Uses Camarilla structure + volume spike + regime filter for edge in both bull and bear.
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
    
    # Get 1d data for Camarilla levels, trend filter, and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Calculate daily EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    hl_range_1d = high_1d - low_1d
    r1_1d = close_1d + (1.1 * hl_range_1d / 12)  # R1 = close + 1.1*(high-low)/12
    s1_1d = close_1d - (1.1 * hl_range_1d / 12)  # S1 = close - 1.1*(high-low)/12
    
    # Align Camarilla levels to match prices index
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate choppiness index on 1d for regime filter
    # CHOP = 100 * LOG10(SUM(ATR(14)) / (LOG10(HIGHEST HIGH(14) - LOWEST LOW(14))) / LOG10(14))
    # Simplified: CHOP = 100 * LOG10(ATR_sum / (HH - LL)) / LOG10(14)
    tr_1d = np.maximum(np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1])), np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with index
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = highest_high_14 - lowest_low_14
    
    # Avoid division by zero
    chop_ratio = np.where(chop_denominator > 0, atr_14_1d / chop_denominator, np.nan)
    chop_1d = 100 * (np.log10(chop_ratio) / np.log10(14))
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), volume MA (20), ATR (14), HH/LL (14)
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
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
        # Regime filter: chop < 38.2 = trending, chop > 61.8 = choppy
        is_trending = chop_1d_aligned[i] < 38.2
        is_choppy = chop_1d_aligned[i] > 61.8
        
        if position == 0:
            # Determine entry logic based on regime
            if is_trending:
                # Trending market: breakout in direction of trend
                long_setup = (close[i] > r1_aligned[i]) and htf_1d_bullish and volume_confirm
                short_setup = (close[i] < s1_aligned[i]) and htf_1d_bearish and volume_confirm
            elif is_choppy:
                # Choppy market: fade the breakout (mean reversion)
                long_setup = (close[i] < s1_aligned[i]) and htf_1d_bullish and volume_confirm
                short_setup = (close[i] > r1_aligned[i]) and htf_1d_bearish and volume_confirm
            else:
                # Neutral regime: no trades
                long_setup = False
                short_setup = False
            
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
            # Exit conditions
            if is_trending:
                # In trending market: exit on trend reversal or opposite S1 touch
                if (not htf_1d_bullish) or (close[i] <= s1_aligned[i]):
                    signals[i] = 0.0
                    position = 0
            else:
                # In choppy/neutral: exit on opposite R1 touch (mean reversion target) or chop exit
                if (close[i] >= r1_aligned[i]) or (not is_choppy):
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if is_trending:
                # In trending market: exit on trend reversal or opposite R1 touch
                if (htf_1d_bullish) or (close[i] >= r1_aligned[i]):
                    signals[i] = 0.0
                    position = 0
            else:
                # In choppy/neutral: exit on opposite S1 touch (mean reversion target) or chop exit
                if (close[i] <= s1_aligned[i]) or (not is_choppy):
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeRegime_v1"
timeframe = "4h"
leverage = 1.0