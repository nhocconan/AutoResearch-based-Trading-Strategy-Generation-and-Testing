#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d ATR-based volume spike and 1d EMA34 trend filter
# Long when price breaks above 4h Camarilla R3 level AND 1d volume > 1.5x 20-period ATR-scaled average AND close > 1d EMA34
# Short when price breaks below 4h Camarilla S3 level AND 1d volume > 1.5x 20-period ATR-scaled average AND close < 1d EMA34
# Exit when price crosses 4h Camarilla pivot point (mean reversion)
# Uses ATR-scaled volume filter to adapt to volatility regimes, reducing false breakouts in low vol
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_1dATRVolume_1dEMA34"
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
    
    # Get 1d data ONCE before loop for volume and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(20) for volume spike filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first TR
    atr_20_1d = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d volume average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR-scaled volume filter: volume > 1.5x (volume MA * ATR/price) to normalize for volatility
    price_1d = close_1d
    atr_ratio = atr_20_1d / price_1d
    vol_threshold = vol_ma_20 * (1.0 + 1.5 * atr_ratio)  # dynamic threshold based on volatility
    volume_filter_1d = vol_1d > vol_threshold
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d filters to 4h timeframe
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data ONCE before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_r3 = close_4h + (1.1 * (high_4h - low_4h) / 2)
    camarilla_s3 = close_4h - (1.1 * (high_4h - low_4h) / 2)
    camarilla_pivot = (high_4h + low_4h + close_4h) / 3  # Standard pivot point
    
    # Align Camarilla levels to 4h timeframe (same df_4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_filter_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND volume spike AND above 1d EMA34
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND volume spike AND below 1d EMA34
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals