#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 level AND price > EMA34(1d) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S3 level AND price < EMA34(1d) AND volume > 2.0x 20-period average
# Exit when price returns to Camarilla pivot point (mean reversion) OR trend flips (price crosses EMA34(1d))
# Camarilla pivots derived from prior 1d range: R3 = pivot + 1.1*(high-low)/2, S3 = pivot - 1.1*(high-low)/2
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend whipsaws in bear markets
# Volume spike confirms institutional participation
# Target: 12-37 trades/year per symbol (50-150 total over 4 years)
# Discrete sizing (0.25) to limit fee drag

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    # Pivot = (high + low + close) / 3
    # R3 = pivot + 1.1*(high - low)/2
    # S3 = pivot - 1.1*(high - low)/2
    pivot_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r3_1d = pivot_1d + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    s3_1d = pivot_1d - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d.values)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d.values)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d.values)
    
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
        if (np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND price > EMA34(1d) AND volume spike
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND price < EMA34(1d) AND volume spike
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot (mean reversion) OR price < EMA34(1d) (trend flip)
            if (close[i] <= pivot_1d_aligned[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla pivot (mean reversion) OR price > EMA34(1d) (trend flip)
            if (close[i] >= pivot_1d_aligned[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals