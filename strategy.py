#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation
Hypothesis: On 4h timeframe, Camarilla pivot R1/S1 level breakouts with 1d EMA34 trend filter and volume confirmation (>1.5x 50-bar avg) capture institutional breakouts in both bull and bear markets. Uses higher timeframe trend filter to reduce noise and avoid lower timeframe whipsaws. Targets 20-50 trades/year to minimize fee drag while maintaining edge via trend and volume confirmation. Works in bull markets via breakouts and bear markets via mean reversion at extreme levels.
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
    
    # Get 1d data for HTF trend and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume average (50-period = ~4.17 days on 4h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(50, 34)  # volume MA, EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_34_val = ema_34_aligned[i]
        camarilla_R1_val = camarilla_R1_aligned[i]
        camarilla_S1_val = camarilla_S1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 1.5x 50-period average
        volume_confirmed = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: break above R1 with uptrend (close > EMA34) and volume confirmation
            long_signal = (high_val > camarilla_R1_val) and (close_val > ema_34_val) and volume_confirmed
            # Short: break below S1 with downtrend (close < EMA34) and volume confirmation
            short_signal = (low_val < camarilla_S1_val) and (close_val < ema_34_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Trend reversal: close crosses below EMA34
            if close_val < ema_34_val:
                signals[i] = 0.0
                position = 0
            # 2. Price returns to Camarilla pivot point (mean reversion)
            elif low_val <= camarilla_S1_val and high_val >= camarilla_R1_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Trend reversal: close crosses above EMA34
            if close_val > ema_34_val:
                signals[i] = 0.0
                position = 0
            # 2. Price returns to Camarilla pivot point (mean reversion)
            elif low_val <= camarilla_S1_val and high_val >= camarilla_R1_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0