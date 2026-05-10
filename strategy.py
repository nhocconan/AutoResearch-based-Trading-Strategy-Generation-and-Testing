# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R3/S3 levels act as key intraday support/resistance. A breakout above R3 or below S3 with 1d trend alignment and volume confirmation indicates strong momentum.
# Uses 1d EMA for trend filter and volume spike for confirmation. Designed for low trade frequency (20-40/year) to minimize fee drag.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # We'll use the 1d data to get daily OHLC
    # Camarilla: Close +- (High-Low) * 1.1/12 for R3/S3
    # R3 = Close + (High-Low) * 1.1/12 * 4
    # S3 = Close - (High-Low) * 1.1/12 * 4
    # Actually: R3 = Close + (High-Low) * 1.1/6, S3 = Close - (High-Low) * 1.1/6
    # Let's compute properly:
    # Typical Camarilla: 
    # R4 = Close + (High-Low) * 1.1/2
    # R3 = Close + (High-Low) * 1.1/4
    # R2 = Close + (High-Low) * 1.1/6
    # R1 = Close + (High-Low) * 1.1/12
    # S1 = Close - (High-Low) * 1.1/12
    # S2 = Close - (High-Low) * 1.1/6
    # S3 = Close - (High-Low) * 1.1/4
    # S4 = Close - (High-Low) * 1.1/2
    # We want R3 and S3: multiplier = 1.1/4
    
    # Get previous day's OHLC (shift by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    # First day has no previous, so we'll skip until we have data
    prev_close[0] = close_1d[0]  # Will be filtered by min_periods anyway
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    
    camarilla_mult = 1.1 / 4  # For R3/S3
    camarilla_range = (prev_high - prev_low) * camarilla_mult
    r3_level = prev_close + camarilla_range
    s3_level = prev_close - camarilla_range
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # Volume confirmation (20-period average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20) + 1  # Need EMA34 and volume MA20
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long breakout: price closes above R3 AND 1d uptrend AND volume confirmation
            if close[i] > r3_aligned[i] and close[i] > ema_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below S3 AND 1d downtrend AND volume confirmation
            elif close[i] < s3_aligned[i] and close[i] < ema_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below EMA (trend change) OR loses volume momentum
            if close[i] < ema_1d_aligned[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above EMA (trend change) OR loses volume momentum
            if close[i] > ema_1d_aligned[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals