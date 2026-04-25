#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R1 level with 1d uptrend (price > EMA34) and volume spike (>2.0x avg); enter short when price breaks below S1 level with 1d downtrend (price < EMA34) and volume spike. Exit on opposite Camarilla level touch (S1 for long, R1 for short) or trend reversal. Uses discrete sizing (0.25) to minimize fee churn. Target: 20-40 trades/year by requiring tight confluence of Camarilla breakout, 1d trend alignment, and volume confirmation. Fixed: use aligned 1d close for trend filter and proper warmup.
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
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C=close, H=high, L=low of previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Align 1d close price for trend filter
    df_1d_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate 4h volume ratio (current vs 24-period average = 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA (34) and volume MA (24)
    start_idx = max(40, 34, 24)  # 40 for Camarilla shift, 34 for EMA, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(df_1d_close_aligned[i])):
            # Hold current position until data is ready
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Determine 1d trend (bullish = price above EMA34)
        htf_1d_bullish = df_1d_close_aligned[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = df_1d_close_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: need significant spike (vol_ratio > 2.0)
        volume_confirmed = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 1d uptrend + volume confirmation
            long_setup = (close[i] > camarilla_r1_aligned[i]) and htf_1d_bullish and volume_confirmed
            
            # Short setup: price breaks below Camarilla S1 + 1d downtrend + volume confirmation
            short_setup = (close[i] < camarilla_s1_aligned[i]) and htf_1d_bearish and volume_confirmed
            
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
            if (close[i] <= camarilla_s1_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 (opposite level) OR 1d trend turns bullish
            if (close[i] >= camarilla_r1_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0