#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 level AND price > EMA50(1w) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S3 level AND price < EMA50(1w) AND volume > 2.0x 20-period average
# Exit when price returns to Camarilla pivot point (mean reversion) OR trend flips (price crosses EMA50(1w))
# 1w EMA50 provides higher timeframe trend filter to avoid counter-trend whipsaws in bear markets
# Volume spike confirms institutional participation
# Target: 7-25 trades/year per symbol (30-100 total over 4 years)
# Discrete sizing (0.25) to limit fee drag

name = "1d_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter and Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels from prior 1w bar
    # Pivot = (high + low + close) / 3
    # R3 = pivot + 1.1*(high - low)/2
    # S3 = pivot - 1.1*(high - low)/2
    pivot_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    r3_1w = pivot_1w + 1.1 * (df_1w['high'] - df_1w['low']) / 2
    s3_1w = pivot_1w - 1.1 * (df_1w['high'] - df_1w['low']) / 2
    
    # Align Camarilla levels to 1d timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w.values)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w.values)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w.values)
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND price > EMA50(1w) AND volume spike
            if (close[i] > r3_1w_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND price < EMA50(1w) AND volume spike
            elif (close[i] < s3_1w_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot (mean reversion) OR price < EMA50(1w) (trend flip)
            if (close[i] <= pivot_1w_aligned[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla pivot (mean reversion) OR price > EMA50(1w) (trend flip)
            if (close[i] >= pivot_1w_aligned[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals