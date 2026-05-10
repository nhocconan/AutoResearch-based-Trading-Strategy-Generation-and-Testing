#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dVWAP_Trend_Volume
# Hypothesis: Uses Camarilla pivot levels (R3/S3) from daily pivot points with volume confirmation and 1d VWAP trend filter.
# Enters long on break above R3 with volume spike and price above 1d VWAP (uptrend).
# Enters short on break below S3 with volume spike and price below 1d VWAP (downtrend).
# Exits when price returns to the daily VWAP or reverses past the opposite Camarilla level.
# Designed for 20-35 trades/year on 4h to avoid overtrading and work in both bull and bear markets.
# Uses Camarilla levels for institutional reference points and volume for confirmation.

name = "4h_Camarilla_R3_S3_Breakout_1dVWAP_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily VWAP calculation (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    # Avoid division by zero
    vwap = np.where(vwap_den == 0, typical_price, vwap)
    
    # Daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Align 1d VWAP to 4h timeframe
    # Calculate VWAP for each 1d bar
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_num_1d = np.cumsum(typical_price_1d * df_1d['volume'].values)
    vwap_den_1d = np.cumsum(df_1d['volume'].values)
    vwap_1d = vwap_num_1d / vwap_den_1d
    vwap_1d = np.where(vwap_den_1d == 0, typical_price_1d, vwap_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume spike detection: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and above 1d VWAP (uptrend)
            if close[i] > camarilla_r3_aligned[i] and volume_spike[i] and close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike and below 1d VWAP (downtrend)
            elif close[i] < camarilla_s3_aligned[i] and volume_spike[i] and close[i] < vwap_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to VWAP or breaks below S3 (reversal signal)
            if close[i] <= vwap_1d_aligned[i] or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to VWAP or breaks above R3 (reversal signal)
            if close[i] >= vwap_1d_aligned[i] or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals