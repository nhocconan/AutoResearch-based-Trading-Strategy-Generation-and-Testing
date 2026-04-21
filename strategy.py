#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Daily Camarilla pivot R1/S1 breakout with weekly EMA34 trend filter and volume spike confirmation. Designed for low trade frequency (~15-25/year) to minimize fee drag and work in both bull/bear markets via trend filter. Uses 1d primary timeframe with 1w HTF for trend context.
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
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # === 1w trend filter: 34-period EMA ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Camarilla pivot levels (based on previous day) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Shift to use previous day's OHLC for today's Camarilla levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar uses current
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_val * 1.1 / 12)
    S1 = pivot - (range_val * 1.1 / 12)
    
    # === Daily volume average (20-period) for spike detection ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma[np.isnan(vol_ma)] = 1.0  # avoid division by zero
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after volume MA warmup
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(R1[i]) or np.isnan(S1[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1w = ema_34_1w_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 2.0 + price above weekly EMA34
            if price_high > R1[i] and vol_spike > 2.0 and price_close > trend_1w:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike > 2.0 + price below weekly EMA34
            elif price_low < S1[i] and vol_spike > 2.0 and price_close < trend_1w:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit on opposite Camarilla level touch or volume spike failure
            if position == 1:
                # Exit long if price touches S1 or volume drops
                if price_low <= S1[i] or vol_spike < 1.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if price touches R1 or volume drops
                if price_high >= R1[i] or vol_spike < 1.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0