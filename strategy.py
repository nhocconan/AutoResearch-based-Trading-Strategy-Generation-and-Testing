#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above R3 AND close > 12h EMA50 AND volume > 2x 20-bar avg
# Short when price breaks below S3 AND close < 12h EMA50 AND volume > 2x 20-bar avg
# Exit when price reverts to Camarilla H3/L3 level or volume drops below average
# Works in both bull and bear markets by only trading breakouts with trend and volume confirmation
# Target: 12-37 trades/year via tight entry conditions reducing whipsaw

name = "6h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3/L3, R3/S3, R4/S4
    # Based on previous day's range
    diff = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * diff / 6  # H3 level
    camarilla_l3 = close_1d - 1.1 * diff / 6  # L3 level
    camarilla_r3 = close_1d + 1.1 * diff / 4  # R3 level
    camarilla_s3 = close_1d - 1.1 * diff / 4  # S3 level
    camarilla_r4 = close_1d + 1.1 * diff / 2  # R4 level
    camarilla_s4 = close_1d - 1.1 * diff / 2  # S4 level
    
    # Prepend zero for alignment (since we use previous day's data)
    camarilla_h3 = np.concatenate([np.array([np.nan]), camarilla_h3[:-1]])
    camarilla_l3 = np.concatenate([np.array([np.nan]), camarilla_l3[:-1]])
    camarilla_r3 = np.concatenate([np.array([np.nan]), camarilla_r3[:-1]])
    camarilla_s3 = np.concatenate([np.array([np.nan]), camarilla_s3[:-1]])
    camarilla_r4 = np.concatenate([np.array([np.nan]), camarilla_r4[:-1]])
    camarilla_s4 = np.concatenate([np.array([np.nan]), camarilla_s4[:-1]])
    
    # Align Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_spike[i]
        ema_trend = ema_50_12h_aligned[i]
        price = close[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND trend up AND volume spike
            if price > r3 and price > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND trend down AND volume spike
            elif price < s3 and price < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price reverts to H3 or trend changes or no volume spike
            if price < h3 or price < ema_trend or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price reverts to L3 or trend changes or no volume spike
            if price > l3 or price > ema_trend or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals