#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d high, low, close for Camarilla pivot
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # 1d Camarilla levels
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
    
    # Align 1d Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume filter: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
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
        
        # Volume confirmation: moderate threshold for selectivity
        volume_confirmed = volume_current > 2.0 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.005 * price_close  # ATR > 0.5% of price
        
        # Long conditions: price breaks below S3 (oversold) with volume and vol filter
        long_signal = volume_confirmed and vol_filter and (price_low < s3_6h[i])
        
        # Short conditions: price breaks above R3 (overbought) with volume and vol filter
        short_signal = volume_confirmed and vol_filter and (price_high > r3_6h[i])
        
        # Exit when price returns to 1d pivot level
        pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
        exit_long = position == 1 and price_close > pivot_6h[i]
        exit_short = position == -1 and price_close < pivot_6h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
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

# Hypothesis: 1d Camarilla levels act as strong support/resistance for 6h price action.
# Enters long when 6h price breaks below S3 (oversold bounce) with volume confirmation (>2.0x average),
# and sufficient volatility (ATR > 0.5% of price).
# Enters short when price breaks above R3 (overbought rejection) with same conditions.
# Exits when price returns to 1d pivot level, capturing mean reversion.
# Uses selective volume filter (2.0x) and volatility filter to reduce trades to ~12-30/year.
# Works in both bull (buying dips) and bear (selling rallies) markets by fading extremes.