#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Designed for low trade frequency (~12-37/year) to minimize fee drag. Works in bull/bear via 1d trend filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # === 12h Camarilla levels (requires 5-period lookback) ===
        if i < 5:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Calculate Camarilla from previous 12h bar (i-1)
        high_prev = prices['high'].iloc[i-1]
        low_prev = prices['low'].iloc[i-1]
        close_prev = prices['close'].iloc[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Camarilla R1 and S1 levels
        R1 = close_prev + range_prev * 1.1 / 12
        S1 = close_prev - range_prev * 1.1 / 12
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 2.0 + price above 1d EMA34
            if price_close > R1 and vol_spike > 2.0 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike > 2.0 + price below 1d EMA34
            elif price_close < S1 and vol_spike > 2.0 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit on opposite Camarilla level touch or trend reversal
            if position == 1:
                # Exit long if price touches S1 or closes below 1d EMA34
                if price_low <= S1 or price_close < trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if price touches R1 or closes above 1d EMA34
                if price_high >= R1 or price_close > trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0