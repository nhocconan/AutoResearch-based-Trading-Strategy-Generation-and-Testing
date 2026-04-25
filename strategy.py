#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_Regime
Hypothesis: Use 12h timeframe with Camarilla H3/L3 breakout confirmed by 1d EMA34 trend and volume spike.
Targets 15-25 trades/year on 12h timeframe to minimize fee drag while capturing strong trends in both bull and bear markets.
Uses discrete position sizing (0.25) to reduce churn and improve test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels (H3, L3) from previous day
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: H3 = close + (high - low) * 1.1/4, L3 = close - (high - low) * 1.1/4
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d Camarilla, 1d EMA34, volume MA
    start_idx = max(1, 34, 20)  # +1 for Camarilla calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price closes above H3 + 1d uptrend + volume spike
            long_setup = (close[i] > camarilla_h3_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            # Short: price closes below L3 + 1d downtrend + volume spike
            short_setup = (close[i] < camarilla_l3_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike[i]
            
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
            # Exit: price closes below L3 OR 1d trend turns down
            if (close[i] < camarilla_l3_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above H3 OR 1d trend turns up
            if (close[i] > camarilla_h3_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0