#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX25 regime filter and volume confirmation
# Long when price breaks above R3 AND ADX > 25 (trending) AND volume > 1.5x 20-bar avg
# Short when price breaks below S3 AND ADX > 25 AND volume confirmation
# Exits when price reverts to R2/S2 levels or ADX < 20 (range) or volume drops
# Target: 12-37 trades/year via tight entry conditions and regime filter reducing whipsaw
# Works in both bull and bear markets by only trading when ADX confirms trending conditions

name = "12h_Camarilla_R3_S3_Breakout_1dADX25_Regime_VolumeFilter_v1"
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
    open_price = prices['open'].values
    
    # Get 1d data for ADX calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smoothed DM
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Prepend zeros for alignment (since we lost bars in calculations)
    adx = np.concatenate([np.full(27, np.nan), adx])  # 14 (TR) + 14 (ADX smoothing) - 1
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Standard Camarilla: 
    # R4 = close + ((high-low) * 1.1/2)
    # R3 = close + ((high-low) * 1.1/4)
    # R2 = close + ((high-low) * 1.1/6)
    # R1 = close + ((high-low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high-low) * 1.1/12)
    # S2 = close - ((high-low) * 1.1/6)
    # S3 = close - ((high-low) * 1.1/4)
    # S4 = close - ((high-low) * 1.1/2)
    
    # We need previous day's OHLC for today's Camarilla levels
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    camarilla_r3_1d = prev_close_1d + ((prev_high_1d - prev_low_1d) * 1.1 / 4)
    camarilla_s3_1d = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.1 / 4)
    camarilla_r2_1d = prev_close_1d + ((prev_high_1d - prev_low_1d) * 1.1 / 6)
    camarilla_s2_1d = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.1 / 6)
    
    # Align Camarilla levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2_1d)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or 
            np.isnan(s2_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        adx_val = adx_aligned[i]
        curr_close = close[i]
        curr_open = open_price[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND ADX > 25 (trending) AND volume confirmation
            if curr_close > r3 and adx_val > 25 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND ADX > 25 AND volume confirmation
            elif curr_close < s3 and adx_val > 25 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price reverts to R2 or ADX < 20 (range) or no volume
            if curr_close <= r2 or adx_val < 20 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price reverts to S2 or ADX < 20 (range) or no volume
            if curr_close >= s2 or adx_val < 20 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals