#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
# Uses Camarilla pivot levels from 1d timeframe to identify key support/resistance.
# Breakouts above R1 or below S1 are traded in the direction of 1d EMA34 trend.
# Volume confirmation ensures breakouts have sufficient participation.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in both bull (buy R1 breakout with uptrend) and bear (sell S1 breakdown with downtrend).
# Discrete position sizing (0.25) minimizes fee churn while maintaining exposure.

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:  # Need at least 34 for EMA + buffer
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    # Camarilla: based on previous day's high, low, close
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume on 12h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(40, 20) + 1  # 41 (for EMA34 and volume MA20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Camarilla breakout conditions
        breakout_r1 = curr_close > camarilla_r1_aligned[i]  # Break above R1
        breakdown_s1 = curr_close < camarilla_s1_aligned[i]  # Break below S1
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R1 AND uptrend AND volume confirmation
            if breakout_r1 and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 AND downtrend AND volume confirmation
            elif breakdown_s1 and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below S1 (reversal signal)
            if curr_close < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above R1 (reversal signal)
            if curr_close > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals