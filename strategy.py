#!/usr/bin/env python3
# 6h_1d_1w_camarilla_pivot_volume_v1
# Strategy: 6h Camarilla pivot breakout with daily/weekly pivot direction and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) act as institutional breakout/reversal zones.
# In trending markets: breakouts above R4 or below S4 with volume confirmation continue the trend.
# In ranging markets: reversals at R3/S3 with volume exhaustion provide mean reversion.
# Weekly pivot adds higher-timeframe bias: only take long signals when price > weekly pivot,
# only short when price < weekly pivot. Designed for low trade frequency (~15-25/year) to
# minimize fee drag while capturing sustained moves in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Using previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day values (shift by 1 to avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = high_1d[0]  # First day uses same day
    low_1d_prev[0] = low_1d[0]
    close_1d_prev[0] = close_1d[0]
    
    # Calculate pivot point
    pivot_1d = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    range_1d = high_1d_prev - low_1d_prev
    
    # Camarilla levels
    r3_1d = pivot_1d + range_1d * 1.1 / 2.0
    s3_1d = pivot_1d - range_1d * 1.1 / 2.0
    r4_1d = pivot_1d + range_1d * 1.1
    s4_1d = pivot_1d - range_1d * 1.1
    
    # Align 1d Camarilla levels to 6h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 1w pivot for trend bias
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week values
    high_1w_prev = np.roll(high_1w, 1)
    low_1w_prev = np.roll(low_1w, 1)
    close_1w_prev = np.roll(close_1w, 1)
    high_1w_prev[0] = high_1w[0]
    low_1w_prev[0] = low_1w[0]
    close_1w_prev[0] = close_1w[0]
    
    pivot_1w = (high_1w_prev + low_1w_prev + close_1w_prev) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # 6h ATR for volatility filter (avoid extreme volatility)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or \
           np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or \
           np.isnan(pivot_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i]) or \
           np.isnan(vol_ma[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: avoid extreme volatility ( ATR > 2.5x MA )
        vol_filter = atr[i] <= 2.5 * atr_ma[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Price levels
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        weekly_pivot = pivot_1w_aligned[i]
        
        # Determine market bias from weekly pivot
        long_bias = close[i] > weekly_pivot
        short_bias = close[i] < weekly_pivot
        
        # Entry conditions
        # Long: Breakout above R4 with volume and bias OR reversal at S3 with volume exhaustion
        long_breakout = close[i] > r4 and close[i-1] <= r4 and vol_confirm and long_bias and vol_filter
        long_reversal = close[i] > s3 and close[i-1] <= s3 and vol_confirm and not vol_filter  # reversal on volume exhaustion
        
        # Short: Breakdown below S4 with volume and bias OR reversal at R3 with volume exhaustion
        short_breakout = close[i] < s4 and close[i-1] >= s4 and vol_confirm and short_bias and vol_filter
        short_reversal = close[i] < r3 and close[i-1] >= r3 and vol_confirm and not vol_filter
        
        # Exit conditions: reverse signal or volatility spike
        exit_long = (close[i] < s3 and vol_confirm) or (not vol_filter and position == 1)
        exit_short = (close[i] > r3 and vol_confirm) or (not vol_filter and position == -1)
        
        if (long_breakout or long_reversal) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (short_breakout or short_reversal) and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals