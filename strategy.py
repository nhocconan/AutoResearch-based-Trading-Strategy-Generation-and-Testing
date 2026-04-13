#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using 12h Camarilla pivot levels for mean reversion
    # Fade at R3/S3 levels with volume confirmation and 1d trend filter
    # Long: price <= S3 AND 1d close > 1d EMA50 AND volume > 1.2x avg
    # Short: price >= R3 AND 1d close < 1d EMA50 AND volume > 1.2x avg
    # Exit: price crosses Camarilla H3/L3 levels or volume dry-up
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (using typical price)
    typical_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Camarilla levels calculation
    camarilla_h5 = np.full(len(typical_12h), np.nan)
    camarilla_h4 = np.full(len(typical_12h), np.nan)
    camarilla_h3 = np.full(len(typical_12h), np.nan)
    camarilla_l3 = np.full(len(typical_12h), np.nan)
    camarilla_l4 = np.full(len(typical_12h), np.nan)
    camarilla_l5 = np.full(len(typical_12h), np.nan)
    
    for i in range(len(typical_12h)):
        if i >= 1:  # Need previous bar for calculation
            rng = h_12h[i-1] - l_12h[i-1]
            camarilla_h5[i] = c_12h[i-1] + rng * 1.5
            camarilla_h4[i] = c_12h[i-1] + rng * 1.25
            camarilla_h3[i] = c_12h[i-1] + rng * 1.125
            camarilla_l3[i] = c_12h[i-1] - rng * 1.125
            camarilla_l4[i] = c_12h[i-1] - rng * 1.25
            camarilla_l5[i] = c_12h[i-1] - rng * 1.5
    
    # Align Camarilla levels to 6h
    h5_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h5)
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    l5_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l5)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    multiplier = 2 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_1d[i] = close_1d[i]
        elif not np.isnan(ema_1d[i-1]):
            ema_1d[i] = (close_1d[i] - ema_1d[i-1]) * multiplier + ema_1d[i-1]
    
    # Align 1d EMA50 to 6h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 6h volume for confirmation (>1.2x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_12h_aligned[i]) or np.isnan(l3_12h_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d close > EMA50 = bullish bias, < EMA50 = bearish bias
        bullish_bias = close[i] > ema_1d_aligned[i]
        bearish_bias = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Camarilla S3/R3 fade with trend and volume
        long_entry = (close[i] <= l3_12h_aligned[i]) and bullish_bias and vol_confirm
        short_entry = (close[i] >= h3_12h_aligned[i]) and bearish_bias and vol_confirm
        
        # Exit logic: price crosses H3/L3 levels or volume dry-up
        long_exit = (close[i] >= h3_12h_aligned[i]) or not vol_confirm
        short_exit = (close[i] <= l3_12h_aligned[i]) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_camarilla_fade_v1"
timeframe = "6h"
leverage = 1.0