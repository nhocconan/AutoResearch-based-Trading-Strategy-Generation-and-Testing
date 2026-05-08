#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2"
timezone = "UTC"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Only calculate if we have previous day data
    valid_prev = ~(np.isnan(high_prev) | np.isnan(low_prev) | np.isnan(close_prev))
    
    # Camarilla R3 and S3 levels
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    camarilla_range = (high_prev - low_prev) * 1.1 / 2.0
    r3 = close_prev + camarilla_range
    s3 = close_prev - camarilla_range
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily trend filter: EMA 34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly volatility regime: ATR(14) normalized by price
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_percent = atr_14 / np.where(close_1w > 0, close_1w, 1.0)
    
    # Align ATR percentage to 12h timeframe
    atr_percent_aligned = align_htf_to_ltf(prices, df_1w, atr_percent)
    
    # Volume confirmation: volume ratio to 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_percent_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly volatility filter: only trade when volatility is elevated (above median)
        vol_filter = atr_percent_aligned[i] > 0.015  # 1.5% ATR as percentage of price
        
        if position == 0 and vol_filter:
            # Long: price breaks above R3 + above daily EMA34 + volume confirmation
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + below daily EMA34 + volume confirmation
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or below daily EMA34
            if close[i] < s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or above daily EMA34
            if close[i] > r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals