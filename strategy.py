#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeConfirm_v1
Hypothesis: Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation. Designed for low trade frequency (~15-25/year) to minimize fee drag. Uses 6h primary timeframe with 12h HTF for trend and volume context. Works in bull/bear by requiring alignment with 12h trend and volume spike to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h trend filter: 50-period EMA ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 12h volume average (20-period) for spike detection ===
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h[np.isnan(vol_ma_12h)] = 1.0  # avoid division by zero
    vol_ratio_12h = volume_12h / vol_ma_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ratio_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla levels from previous 6h bar
        if i == 0:
            continue
        high_prev = prices['high'].iloc[i-1]
        low_prev = prices['low'].iloc[i-1]
        close_prev = prices['close'].iloc[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for current 6h bar
        r3 = close_prev + range_prev * 1.1 / 4
        s3 = close_prev - range_prev * 1.1 / 4
        r4 = close_prev + range_prev * 1.1 / 2
        s4 = close_prev - range_prev * 1.1 / 2
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        trend_12h = ema_50_12h_aligned[i]
        vol_spike = vol_ratio_12h_aligned[i]
        
        if position == 0:
            # Long: price closes above R3 + volume spike > 2.0 + price above 12h EMA50
            if price_close > r3 and vol_spike > 2.0 and price_close > trend_12h:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price closes below S3 + volume spike > 2.0 + price below 12h EMA50
            elif price_close < s3 and vol_spike > 2.0 and price_close < trend_12h:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit conditions: opposite Camarilla level touch or trend reversal
            if position == 1:
                # Exit long if price touches S3 or trend turns bearish
                if price_low <= s3 or price_close < trend_12h:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if price touches R3 or trend turns bullish
                if price_high >= r3 or price_close > trend_12h:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0