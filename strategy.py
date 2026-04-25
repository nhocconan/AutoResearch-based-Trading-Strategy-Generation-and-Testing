#!/usr/bin/env python3
"""
6h_KeltnerBreakout_VolumeSpike_1dTrend_v1
Hypothesis: Trade Keltner Channel breakouts with volume spike confirmation and 1d EMA trend filter.
Long when price breaks above upper Keltner (EMA20 + 2*ATR10) with volume > 1.5x average volume and price > 1d EMA50.
Short when price breaks below lower Keltner (EMA20 - 2*ATR10) with volume > 1.5x average volume and price < 1d EMA50.
Exit on opposite Keltner touch or trend reversal.
Uses 6h timeframe for entries with 1d HTF trend filter.
Position size: 0.25 to balance drawdown and return.
Target: 15-30 trades/year to stay well under 300-trade 6h hard max.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Keltner Channel components on 6h data
    # EMA20 for middle line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR10 for channel width
    tr1 = np.maximum(high - low, 0)
    tr2 = np.absolute(high - np.roll(close, 1))
    tr3 = np.absolute(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Upper and lower Keltner bands
    upper_keltner = ema_20 + 2.0 * atr_10
    lower_keltner = ema_20 - 2.0 * atr_10
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA20 (20) and ATR10 (10) and volume MA (20)
    start_idx = max(20, 10, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20[i]) or np.isnan(atr_10[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long setup: price breaks above upper Keltner + volume spike + 1d uptrend
            long_setup = (close[i] > upper_keltner[i]) and volume_spike and htf_1d_bullish
            
            # Short setup: price breaks below lower Keltner + volume spike + 1d downtrend
            short_setup = (close[i] < lower_keltner[i]) and volume_spike and htf_1d_bearish
            
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
            # Exit: price touches lower Keltner (stop) OR 1d trend turns bearish
            if (close[i] <= lower_keltner[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches upper Keltner (stop) OR 1d trend turns bullish
            if (close[i] >= upper_keltner[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_KeltnerBreakout_VolumeSpike_1dTrend_v1"
timeframe = "6h"
leverage = 1.0