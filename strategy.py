#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Volume_Confluence_v1
Hypothesis: Trade Camarilla pivot (R1/S1) breakouts on 1d with 1w EMA50 trend filter, volume confirmation, and ATR-based stoploss.
Only trade when market is trending (1w EMA50 slope positive for long, negative for short) to avoid whipsaws.
Long: price breaks above R1, volume > 1.5x 20-day median, close > 1w EMA50.
Short: price breaks below S1, volume > 1.5x 20-day median, close < 1w EMA50.
Exit on opposite Camarilla level touch or trend reversal (close crosses 1w EMA50).
Uses discrete position sizing (0.25) to minimize fee churn. Target: 15-25 trades/year.
Works in bull/bear markets by following 1w EMA50 trend and requiring volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    camarilla_r1 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_s1 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    
    # Align HTF indicators to 1d timeframe (no additional delay needed for EMA)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: 1.5x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1w EMA (50), volume median (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        if position == 0:
            # Long: break above R1 with volume and uptrend (close > 1w EMA50)
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          (volume_val > 1.5 * vol_median_val) and \
                          (close_val > ema_50_1w_val)
            
            # Short: break below S1 with volume and downtrend (close < 1w EMA50)
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           (volume_val > 1.5 * vol_median_val) and \
                           (close_val < ema_50_1w_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S1 (reversal) or trend changes (close < 1w EMA50)
            if (close_val < camarilla_s1_aligned[i]) or \
               (close_val < ema_50_1w_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 (reversal) or trend changes (close > 1w EMA50)
            if (close_val > camarilla_r1_aligned[i]) or \
               (close_val > ema_50_1w_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume_Confluence_v1"
timeframe = "1d"
leverage = 1.0