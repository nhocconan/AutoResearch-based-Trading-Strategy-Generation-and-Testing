#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Camarilla pivot levels with volume confirmation and ADX trend filter.
# Camarilla levels (R3, S3) act as strong support/resistance in trending markets.
# Long when price breaks above R3 in an uptrend with volume confirmation.
# Short when price breaks below S3 in a downtrend with volume confirmation.
# Uses daily ADX(14) > 25 to filter for trending conditions only.
# Designed for low trade frequency (20-50/year) to minimize fee drag and capture high-probability breakouts.

name = "4h_Camarilla_R3S3_Breakout_ADX_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    camarilla_r3 = np.zeros_like(close_1d)  # R3 level
    camarilla_s3 = np.zeros_like(close_1d)  # S3 level
    
    for i in range(1, len(close_1d)):
        # Previous day's high, low, close
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Range
        rng = ph - pl
        
        # Camarilla levels
        camarilla_r3[i] = pc + (rng * 1.1/2)  # R3 = C + (H-L)*1.1/2
        camarilla_s3[i] = pc - (rng * 1.1/2)  # S3 = C - (H-L)*1.1/2
    
    # First day has no prior data
    camarilla_r3[0] = camarilla_s3[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get daily ADX(14) for trend filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])  # First TR is 0
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 14)
    di_plus = wilders_smooth(dm_plus, 14)
    di_minus = wilders_smooth(dm_minus, 14)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smooth(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above R3 in uptrend (ADX > 25) with volume
            if (adx_aligned[i] > 25 and  # Trending condition
                close[i] > camarilla_r3_aligned[i] and  # Break above R3
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: break below S3 in downtrend (ADX > 25) with volume
            elif (adx_aligned[i] > 25 and  # Trending condition
                  close[i] < camarilla_s3_aligned[i] and  # Break below S3
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below S3 or trend weakens (ADX < 20)
            if close[i] < camarilla_s3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above R3 or trend weakens (ADX < 20)
            if close[i] > camarilla_r3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals