#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and 1w EMA34 trend filter
# Long when price breaks above Camarilla R3 level (1d) AND 1d volume > 2.0x 20-period average AND close > 1w EMA34
# Short when price breaks below Camarilla S3 level (1d) AND 1d volume > 2.0x 20-period average AND close < 1w EMA34
# Exit when price retouches Camarilla pivot point (mean reversion to equilibrium)
# Uses 12h primary timeframe with 1d HTF for Camarilla levels/volume and 1w HTF for trend filter
# Camarilla levels provide institutional support/resistance; volume confirms conviction; EMA filter avoids counter-trend trades
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_Camarilla_R3S3_Breakout_1dVolume_1wEMA34"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla levels and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3, pivot)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_range = high_1d - low_1d
    camarilla_r3 = camarilla_pivot + (camarilla_range * 1.1 / 4)
    camarilla_s3 = camarilla_pivot - (camarilla_range * 1.1 / 4)
    
    # Calculate 1d volume spike filter
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_filter_1d = vol_1d > (2.0 * vol_ma_20)
    else:
        volume_filter_1d = np.zeros(len(df_1d), dtype=bool)
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    # Align 1w EMA34 to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND volume spike AND above 1w EMA34
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter_1d_aligned[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND volume spike AND below 1w EMA34
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter_1d_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla pivot point (mean reversion)
            if close[i] <= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla pivot point (mean reversion)
            if close[i] >= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals