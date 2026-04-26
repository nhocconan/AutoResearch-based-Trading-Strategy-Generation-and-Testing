#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Use 1d EMA34 trend filter with Camarilla R1/S1 breakout and volume spike confirmation.
Long when: price > 1d EMA34 + break above Camarilla R1 + volume > 1.5x 20-bar avg.
Short when: price < 1d EMA34 + break below Camarilla S1 + volume > 1.5x 20-bar avg.
Exit when: price reverts to Camarilla midpoint (H5/L5) or 1d trend reverses.
Designed for BTC/ETH: captures intraday breakouts aligned with daily trend, avoids counter-trend trades.
Target: 20-40 trades/year for optimal fee efficiency and test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA34 trend (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's Camarilla levels (H4, L4, H5, L5)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_h4 = high_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = low_1d - 1.1 * (high_1d - low_1d) / 2
    camarilla_h5 = camarilla_h4 + 1.1 * (high_1d - low_1d) / 2
    camarilla_l5 = camarilla_l4 - 1.1 * (high_1d - low_1d) / 2
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_mid = (camarilla_h4_aligned + camarilla_l4_aligned) / 2  # H5/L5 midpoint
    
    # Volume confirmation (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 34 for EMA, 20 for volume MA
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_h5_aligned[i]) or 
            np.isnan(camarilla_l5_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_val = volume[i]
        size = 0.25
        
        if position == 0:
            # Flat - look for breakout in direction of 1d trend
            if ema_34_1d_aligned[i] > 0:  # Valid EMA value
                # Long: price above 1d EMA + break above Camarilla H5 + volume spike
                long_cond = (close_val > ema_34_1d_aligned[i]) and \
                            (close_val > camarilla_h5_aligned[i]) and \
                            (vol_val > 1.5 * vol_ma[i])
                # Short: price below 1d EMA + break below Camarilla L5 + volume spike
                short_cond = (close_val < ema_34_1d_aligned[i]) and \
                             (close_val < camarilla_l5_aligned[i]) and \
                             (vol_val > 1.5 * vol_ma[i])
                
                if long_cond:
                    signals[i] = size
                    position = 1
                elif short_cond:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to midpoint or 1d trend turns bearish
            if close_val < camarilla_mid[i] or ema_34_1d_aligned[i] < close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to midpoint or 1d trend turns bullish
            if close_val > camarilla_mid[i] or ema_34_1d_aligned[i] > close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0