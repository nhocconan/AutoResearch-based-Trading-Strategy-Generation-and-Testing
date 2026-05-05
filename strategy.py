#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 AND 1d ADX > 25 (strong trend) AND volume > 2.0x 20-period average
# Short when price breaks below 1d Camarilla S3 AND 1d ADX > 25 (strong trend) AND volume > 2.0x 20-period average
# Exit when price crosses 1d Camarilla midpoint (R3/S3 midpoint) OR ADX < 20 (trend weakening)
# Uses proven Camarilla levels from 1d timeframe for structure, ADX for trend strength filter, and volume for confirmation
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag while maintaining edge in both bull and bear markets
# Timeframe: 6h (primary)
# Target symbols: BTC/ETH/SOL (avoid SOL-only bias)

name = "6h_Camarilla_R3S3_Breakout_1dADX_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (R3, S3, midpoint)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    #          Midpoint (R3/S3) = close
    high_low_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * high_low_1d * 1.1 / 4.0
    camarilla_s3 = close_1d - 1.1 * high_low_1d * 1.1 / 4.0
    camarilla_mid = close_1d  # Midpoint between R3 and S3 is the close
    
    # Calculate 1d ADX(14)
    # ADX calculation: +DM, -DM, TR, +DI, -DI, DX, ADX
    high_shift = np.roll(high_1d, 1)
    low_shift = np.roll(low_1d, 1)
    high_shift[0] = high_1d[0]
    low_shift[0] = low_1d[0]
    
    plus_dm = np.where((high_1d - high_shift) > (low_shift - low_1d), np.maximum(high_1d - high_shift, 0), 0)
    minus_dm = np.where((low_shift - low_1d) > (high_1d - high_shift), np.maximum(low_shift - low_1d, 0), 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    if len(tr) >= period_adx:
        atr = wilders_smoothing(tr, period_adx)
        plus_di = 100 * wilders_smoothing(plus_dm, period_adx) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, period_adx) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smoothing(dx, period_adx)
    else:
        adx = np.full_like(close_1d, np.nan)
    
    # Align HTF indicators to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation on 6h (threshold: 2.0x for strict filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND ADX > 25 (strong trend) AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                adx_aligned[i] > 25.0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND ADX > 25 (strong trend) AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  adx_aligned[i] > 25.0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla midpoint OR ADX < 20 (trend weakening)
            if close[i] < camarilla_mid_aligned[i] or adx_aligned[i] < 20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla midpoint OR ADX < 20 (trend weakening)
            if close[i] > camarilla_mid_aligned[i] or adx_aligned[i] < 20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals