#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d high, low, close for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # 1d Camarilla levels - using correct formulas
    r3_1d = pivot_1d + (range_1d * 1.1 / 4)
    r4_1d = pivot_1d + (range_1d * 1.1 / 2)
    s3_1d = pivot_1d - (range_1d * 1.1 / 4)
    s4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Shift by 1 to use only completed 1d bars
    r3_1d = np.roll(r3_1d, 1)
    r4_1d = np.roll(r4_1d, 1)
    s3_1d = np.roll(s3_1d, 1)
    s4_1d = np.roll(s4_1d, 1)
    r3_1d[0] = np.nan
    r4_1d[0] = np.nan
    s3_1d[0] = np.nan
    s4_1d[0] = np.nan
    
    # Align 1d levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w high, low, close for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w pivot and ranges
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # 1w Camarilla levels - using correct formulas
    r3_1w = pivot_1w + (range_1w * 1.1 / 4)
    r4_1w = pivot_1w + (range_1w * 1.1 / 2)
    s3_1w = pivot_1w - (range_1w * 1.1 / 4)
    s4_1w = pivot_1w - (range_1w * 1.1 / 2)
    
    # Shift by 1 to use only completed 1w bars
    r3_1w = np.roll(r3_1w, 1)
    r4_1w = np.roll(r4_1w, 1)
    s3_1w = np.roll(s3_1w, 1)
    s4_1w = np.roll(s4_1w, 1)
    r3_1w[0] = np.nan
    r4_1w[0] = np.nan
    s3_1w[0] = np.nan
    s4_1w[0] = np.nan
    
    # Align 1w levels to 4h timeframe
    r3_4h_1w = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_4h_1w = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_4h_1w = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_4h_1w = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # 4h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 1.5x 20-period average (selective)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(r4_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(s4_4h[i]) or
            np.isnan(r3_4h_1w) or np.isnan(r4_4h_1w) or np.isnan(s3_4h_1w) or np.isnan(s4_4h_1w) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation: moderate threshold for selectivity
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.005 * price_close  # ATR > 0.5% of price
        
        # Long conditions: price breaks below S3 (oversold) with volume and vol filter
        long_signal = volume_confirmed and vol_filter and (price_low < s3_4h[i]) and (price_low < s3_4h_1w)
        
        # Short conditions: price breaks above R3 (overbought) with volume and vol filter
        short_signal = volume_confirmed and vol_filter and (price_high > r3_4h[i]) and (price_high > r3_4h_1w)
        
        # Exit when price returns to 1d pivot level (mean reversion target)
        pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
        exit_long = position == 1 and price_close > pivot_4h[i]
        exit_short = position == -1 and price_close < pivot_4h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25  # Size: 25%
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1d and 1w Camarilla levels act as strong support/resistance for 4h price action.
# Enters long when 4h price breaks below both 1d S3 and 1w S3 (double oversold) with volume confirmation (>1.5x average),
# sufficient volatility (ATR > 0.5% of price), targeting mean reversion to 1d pivot.
# Enters short when price breaks above both 1d R3 and 1w R3 (double overbought) with same conditions.
# Uses dual timeframe confirmation (1d+1w) to increase signal quality and reduce false breaks.
# Designed to work in both bull (buying dips) and bear (selling rallies) markets by fading extremes.
# Selective volume filter (1.5x) and volatility filter target ~20-30 trades/year.