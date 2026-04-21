#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Camarilla R1/S1 breakout on 12h with 1d EMA34 trend filter and volume spike confirmation. Designed for low trade frequency (~12-25/year) to minimize fee drag and work in both bull/bear markets via trend alignment. Uses 12h primary timeframe with 1d HTF for trend and volume context.
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
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === Camarilla pivot levels (R1, S1) from previous day ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate typical price for Camarilla
    typical_price = (high + low + close) / 3.0
    range_hl = high - low
    
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    camarilla_multiplier = 1.1 / 12.0
    r1 = typical_price + range_hl * camarilla_multiplier
    s1 = typical_price - range_hl * camarilla_multiplier
    
    # Use previous bar's levels to avoid look-ahead
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(r1_prev[i]) or np.isnan(s1_prev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        r1_level = r1_prev[i]
        s1_level = s1_prev[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 2.0 + price above 1d EMA34
            if price_close > r1_level and vol_spike > 2.0 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike > 2.0 + price below 1d EMA34
            elif price_close < s1_level and vol_spike > 2.0 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit on opposite Camarilla level touch or trend failure
            if position == 1:
                # Exit long if price touches S1 or trend fails
                if price_close < s1_level or price_close < trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if price touches R1 or trend fails
                if price_close > r1_level or price_close > trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0