#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v3
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Enters long when price breaks above R1, close > 1d EMA34, and volume > 2.0x 20-period MA.
Enters short when price breaks below S1, close < 1d EMA34, and volume > 2.0x 20-period MA.
Exits when price reverts to Camarilla pivot (PP) or opposite Camarilla level (S1 for long, R1 for short).
Uses 4h primary timeframe to target 20-50 trades/year (75-200 total over 4 years).
Camarilla levels provide precise intraday support/resistance; 1d EMA34 filters counter-trend trades.
Volume spike confirms institutional participation. Works in bull/bear markets by aligning with 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    typical_price = (high + low + close) / 3.0
    range_ = high - low
    PP = typical_price
    R1 = PP + (range_ * 1.1 / 12)
    S1 = PP - (range_ * 1.1 / 12)
    return PP, R1, S1

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (PP, R1, S1)
    camarilla_high = df_1d['high'].values
    camarilla_low = df_1d['low'].values
    camarilla_close = df_1d['close'].values
    camarilla_PP, camarilla_R1, camarilla_S1 = calculate_camarilla(
        camarilla_high, camarilla_low, camarilla_close
    )
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    PP_aligned = align_htf_to_ltf(prices, df_1d, camarilla_PP)
    R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA34, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1, close > 1d EMA34, volume spike
            if (close[i] > R1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, close < 1d EMA34, volume spike
            elif (close[i] < S1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price reverts to PP or breaks below S1 (contrarian exit)
            if (close[i] <= PP_aligned[i] or close[i] < S1_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price reverts to PP or breaks above R1 (contrarian exit)
            if (close[i] >= PP_aligned[i] or close[i] > R1_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0