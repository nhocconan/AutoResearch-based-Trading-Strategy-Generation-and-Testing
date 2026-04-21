#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R1/S1 breakout on 12h with 1d EMA34 trend filter and volume spike confirmation.
Designed for low trade frequency (~12-30/year) to minimize fee drag and work in both bull/bear markets.
Uses 12h primary timeframe with 1d HTF for trend and volume context.
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
    
    # === Camarilla levels on 12h (using previous day's OHLC) ===
    # Camarilla R1 = Close + (High - Low) * 1.1/12
    # Camarilla S1 = Close - (High - Low) * 1.1/12
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Calculate Camarilla levels using previous bar's OHLC to avoid look-ahead
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = high_12h[0]  # first bar uses current values
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    camarilla_range = prev_high - prev_low
    camarilla_r1 = prev_close + camarilla_range * 1.1 / 12
    camarilla_s1 = prev_close - camarilla_range * 1.1 / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close_12h[i]
        price_high = high_12h[i]
        price_low = low_12h[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        r1_level = camarilla_r1[i]
        s1_level = camarilla_s1[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + volume spike > 2.0 + price above 1d EMA34
            if price_close > r1_level and vol_spike > 2.0 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 + volume spike > 2.0 + price below 1d EMA34
            elif price_close < s1_level and vol_spike > 2.0 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or volume dry-up
            if position == 1:
                # Exit long if price breaks below S1 or volume drops below average
                if price_close < s1_level or vol_spike < 0.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if price breaks above R1 or volume drops below average
                if price_close > r1_level or vol_spike < 0.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0