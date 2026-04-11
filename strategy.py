#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
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
    
    # Align 1w levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_6h = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_6h = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # 6h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume filter: volume > 1.5x 20-period average (selective)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation: selective threshold
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.005 * price_close  # ATR > 0.5% of price
        
        # Long conditions: price breaks below S3 (oversold) with volume and vol filter
        long_signal = volume_confirmed and vol_filter and (price_low < s3_6h[i])
        
        # Short conditions: price breaks above R3 (overbought) with volume and vol filter
        short_signal = volume_confirmed and vol_filter and (price_high > r3_6h[i])
        
        # Exit when price returns to 1w pivot level
        pivot_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
        exit_long = position == 1 and price_close > pivot_6h[i]
        exit_short = position == -1 and price_close < pivot_6h[i]
        
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

# Hypothesis: 1w Camarilla levels act as strong support/resistance for 6h price action.
# Enters long when 6h price breaks below S3 (oversold bounce) with volume confirmation (>1.5x average),
# and sufficient volatility (ATR > 0.5% of price).
# Enters short when price breaks above R3 (overbought rejection) with same conditions.
# Exits when price returns to 1w pivot level, capturing mean reversion.
# Uses selective volume filter (1.5x) and volatility filter to reduce trades to ~15-25/year.
# Works in both bull (buying dips) and bear (selling rallies) markets by fading extremes.