#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: 12h Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume spike confirmation. 
Designed for low trade frequency (~15-30/year) to minimize fee drag and improve generalization across bull/bear markets.
Uses 12h primary timeframe with 1w HTF for trend context and 1d HTF for volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1w trend filter: 50-period EMA ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 12h Camarilla pivot levels (R1, S1) ===
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Calculate previous 12h bar's Camarilla levels
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    
    # First bar: use current values (will be filtered by min_periods anyway)
    prev_high[0] = high_12h[0]
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    # Camarilla R1 and S1
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close_12h[i]
        price_high = high_12h[i]
        price_low = low_12h[i]
        trend_1w = ema_50_1w_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        r1_level = r1[i]
        s1_level = s1[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 2.0 + price above 1w EMA50
            if price_close > r1_level and vol_spike > 2.0 and price_close > trend_1w:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike > 2.0 + price below 1w EMA50
            elif price_close < s1_level and vol_spike > 2.0 and price_close < trend_1w:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit logic: reverse signal or loss of trend/volume conditions
            if position == 1:
                # Exit long: price breaks below S1 OR loss of trend/volume
                if price_close < s1_level or price_close < trend_1w or vol_spike < 1.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price breaks above R1 OR loss of trend/volume
                if price_close > r1_level or price_close > trend_1w or vol_spike < 1.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0