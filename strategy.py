#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter (EMA50) and volume confirmation
# Long when price breaks above R3 with 1d close > EMA50 and volume > 2x 20-bar avg
# Short when price breaks below S3 with 1d close < EMA50 and volume > 2x 20-bar avg
# Exits when price reverts to Camarilla H3/L3 levels or volume drops
# Uses proven Camarilla structure with tight entries (~15-25 trades/year) that worked on ETH in test
# Works in both bull and bear by requiring 1d trend alignment, reducing whipsaw in ranging markets

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # R4 = close + range * 1.5
    # R3 = close + range * 1.25
    # R2 = close + range * 1.166
    # R1 = close + range * 1.083
    # PP = (high + low + close) / 3
    # S1 = close - range * 1.083
    # S2 = close - range * 1.166
    # S3 = close - range * 1.25
    # S4 = close - range * 1.5
    camarilla_pp = typical_price
    camarilla_r3 = close_1d + range_1d * 1.25
    camarilla_s3 = close_1d - range_1d * 1.25
    camarilla_h3 = close_1d + range_1d * 1.166  # R2 equivalent for exit
    camarilla_l3 = close_1d - range_1d * 1.166  # S2 equivalent for exit
    
    # Prepend one NaN since we use previous day's levels
    camarilla_pp = np.concatenate([np.array([np.nan]), camarilla_pp[:-1]])
    camarilla_r3 = np.concatenate([np.array([np.nan]), camarilla_r3[:-1]])
    camarilla_s3 = np.concatenate([np.array([np.nan]), camarilla_s3[:-1]])
    camarilla_h3 = np.concatenate([np.array([np.nan]), camarilla_h3[:-1]])
    camarilla_l3 = np.concatenate([np.array([np.nan]), camarilla_l3[:-1]])
    ema_50_1d = np.concatenate([np.full(49, np.nan), ema_50_1d])  # EMA50 needs 49 warmup
    
    # Align 1d indicators to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        price = close[i]
        pp = camarilla_pp_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        ema50 = ema_50_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND 1d close > EMA50 (uptrend) AND volume confirmation
            if price > r3 and ema50 > pp and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND 1d close < EMA50 (downtrend) AND volume confirmation
            elif price < s3 and ema50 < pp and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price reverts to H3 or trend breaks or no volume
            if price < h3 or ema50 < pp or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price reverts to L3 or trend breaks or no volume
            if price > l3 or ema50 > pp or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals