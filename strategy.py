#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h with 1d EMA34 trend filter and volume confirmation (>2x average). Uses discrete sizing (0.25) to limit fee drag. Target: 12-37 trades/year per symbol to survive bear markets and range conditions. Focus on BTC/ETH as primary symbols.
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
    
    # Get 1d data for HTF trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Camarilla levels (based on previous bar's OHLC)
    def calculate_camarilla(high, low, close):
        range_hl = high - low
        r1 = close + (range_hl * 1.1 / 12)
        s1 = close - (range_hl * 1.1 / 12)
        return r1, s1
    
    # Shift by 1 to use previous bar's OHLC (no look-ahead)
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    close_shift[0] = np.nan
    
    camarilla_r1, camarilla_s1 = calculate_camarilla(high_shift, low_shift, close_shift)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(34, 20)  # EMA34 needs 34, Camarilla needs 20 (due to shift)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: strong spike (vol_ratio > 2.0)
        volume_confirmed = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 1d uptrend + volume confirmation
            long_setup = (close[i] > camarilla_r1[i]) and htf_1d_bullish and volume_confirmed
            
            # Short setup: price breaks below Camarilla S1 + 1d downtrend + volume confirmation
            short_setup = (close[i] < camarilla_s1[i]) and htf_1d_bearish and volume_confirmed
            
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
            # Exit: price touches Camarilla S1 (opposite level) OR 1d trend turns bearish
            if (close[i] <= camarilla_s1[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 (opposite level) OR 1d trend turns bullish
            if (close[i] >= camarilla_r1[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0