#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h high, low, close for pivot calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h pivot and ranges
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # 12h Camarilla levels - using correct formulas
    r3_12h = pivot_12h + (range_12h * 1.1 / 4)
    r4_12h = pivot_12h + (range_12h * 1.1 / 2)
    s3_12h = pivot_12h - (range_12h * 1.1 / 4)
    s4_12h = pivot_12h - (range_12h * 1.1 / 2)
    
    # Shift by 1 to use only completed 12h bars
    r3_12h = np.roll(r3_12h, 1)
    r4_12h = np.roll(r4_12h, 1)
    s3_12h = np.roll(s3_12h, 1)
    s4_12h = np.roll(s4_12h, 1)
    r3_12h[0] = np.nan
    r4_12h[0] = np.nan
    s3_12h[0] = np.nan
    s4_12h[0] = np.nan
    
    # Align 12h levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_4h = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_4h = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_4h = align_htf_to_ltf(prices, df_12h, s4_12h)
    
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
    
    for i in range(300, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(r4_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(s4_4h[i]) or
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
        long_signal = volume_confirmed and vol_filter and (price_low < s3_4h[i])
        
        # Short conditions: price breaks above R3 (overbought) with volume and vol filter
        short_signal = volume_confirmed and vol_filter and (price_high > r3_4h[i])
        
        # Exit when price returns to 12h pivot level
        pivot_4h = align_htf_to_ltf(prices, df_12h, pivot_12h)
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

# Hypothesis: 12h Camarilla levels act as strong support/resistance for 4h price action.
# Enters long when 4h price breaks below S3 (oversold bounce) with volume confirmation (>1.5x average),
# and sufficient volatility (ATR > 0.5% of price).
# Enters short when price breaks above R3 (overbought rejection) with same conditions.
# Exits when price returns to 12h pivot level, capturing mean reversion.
# Uses selective volume filter (1.5x) and volatility filter to reduce trades to ~15-25/year.
# Works in both bull (buying dips) and bear (selling rallies) markets by fading extremes.