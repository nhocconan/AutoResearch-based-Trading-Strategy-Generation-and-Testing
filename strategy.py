# 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT
# Hypothesis: Camarilla R3/S3 breakout with 1d EMA trend filter and volume spike works in both bull and bear markets.
# R3/S3 levels represent strong support/resistance; breakouts with volume indicate institutional interest.
# 1d EMA filter ensures trades align with higher-timeframe trend, reducing whipsaws in ranging markets.
# Volume spike confirms breakout validity, avoiding false breakouts in low-volume environments.
# Designed for 4h timeframe with ~20-50 trades/year to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels (using previous day's OHLC)
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla uses previous day's H, L, C
    high_1d = df_1d_ohlc['high'].values
    low_1d = df_1d_ohlc['low'].values
    close_1d_prev = df_1d_ohlc['close'].values
    
    # Typical price for Camarilla calculation
    typical_price = (high_1d + low_1d + close_1d_prev) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3, S3, R4, S4
    # R3 = close + (high - low) * 1.1 / 2
    # S3 = close - (high - low) * 1.1 / 2
    r3 = close_1d_prev + (range_1d * 1.1 / 2)
    s3 = close_1d_prev - (range_1d * 1.1 / 2)
    r4 = close_1d_prev + (range_1d * 1.1)
    s4 = close_1d_prev - (range_1d * 1.1)
    
    # Align Camarilla levels to 4h timeframe (using previous day's close as anchor)
    r3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d_ohlc, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d_ohlc, s4)
    
    # Volume confirmation - 20-period average volume on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Enough for EMA and Camarilla calculation
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume, above 1d EMA34
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio[i] > 2.0):
                # Avoid extreme extension beyond R4
                if close[i] <= r4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below S3 with volume, below 1d EMA34
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio[i] > 2.0):
                # Avoid extreme extension beyond S4
                if close[i] >= s4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or below 1d EMA34
            if close[i] < s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or above 1d EMA34
            if close[i] > r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals