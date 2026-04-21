#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation. Designed for low trade frequency (~20-40/year) to minimize fee drag and improve generalization across bull/bear markets. Uses 4h primary timeframe with 1d HTF for trend and volume context.
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
    
    # === Camarilla pivot levels from previous day (using 1d OHLC) ===
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 1.5 + price above 1d EMA34
            if price_close > r1_level and vol_spike > 1.5 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike > 1.5 + price below 1d EMA34
            elif price_close < s1_level and vol_spike > 1.5 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse signal or loss of trend/volume confirmation
            if position == 1:
                # Exit long: price below S1 or loss of trend
                if price_close < s1_level or price_close < trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price above R1 or loss of trend
                if price_close > r1_level or price_close > trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0