#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrend_VolumeRegime_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 12h EMA50 trend filter, volume confirmation, and choppiness regime filter. 
In bull markets: buy when price breaks above Camarilla R1 and price > 12h EMA50. 
In bear markets: sell when price breaks below Camarilla S1 and price < 12h EMA50. 
Requires volume > 1.3x 20-period average and choppiness index < 61.8 (trending regime) for confirmation. 
Exit on opposite Camarilla level touch or trend reversal. 
Position size: 0.25 to limit drawdown. 
Target: 50-150 total trades over 4 years = 12-37/year. 
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
    
    # Get 12h data for Camarilla levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Calculate 12h EMA50 for HTF trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Calculate Camarilla levels for each 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    hl_range_12h = high_12h - low_12h
    r1_12h = close_12h + (1.1 * hl_range_12h / 12)  # R1 = close + 1.1*(high-low)/12
    s1_12h = close_12h - (1.1 * hl_range_12h / 12)  # S1 = close - 1.1*(high-low)/12
    
    # Align Camarilla levels to match prices index
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Calculate 12h choppiness index for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (max(high,n) - min(low,n)))
    # Simplified: use rolling max/min and ATR approximation
    tr_12h = np.maximum(np.absolute(high_12h[1:] - low_12h[:-1]), 
                        np.absolute(high_12h[1:] - close_12h[:-1]))
    tr_12h = np.concatenate([[np.nan], tr_12h])  # align with index
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    max_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_12h - min_low_12h
    chop_12h = np.where(chop_denom > 0, 
                        100 * np.log10(atr_14_12h * 14 / chop_denom) / np.log10(14), 
                        50)  # default to neutral when denom=0
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50), volume MA (20), ATR (14), chop (14)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(chop_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend (bullish = price above 12h EMA50)
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        # Regime filter: choppiness index < 61.8 (trending regime)
        regime_filter = chop_12h_aligned[i] < 61.8
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 12h uptrend + volume confirmation + trending regime
            long_setup = (close[i] > r1_aligned[i]) and htf_12h_bullish and volume_confirm and regime_filter
            
            # Short setup: price breaks below Camarilla S1 + 12h downtrend + volume confirmation + trending regime
            short_setup = (close[i] < s1_aligned[i]) and htf_12h_bearish and volume_confirm and regime_filter
            
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
            # Exit: price touches Camarilla S1 (stop) OR 12h trend turns bearish OR regime becomes choppy
            if (close[i] <= s1_aligned[i]) or (not htf_12h_bullish) or (not regime_filter):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 (stop) OR 12h trend turns bullish OR regime becomes choppy
            if (close[i] >= r1_aligned[i]) or (htf_12h_bullish) or (not regime_filter):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeRegime_v1"
timeframe = "4h"
leverage = 1.0