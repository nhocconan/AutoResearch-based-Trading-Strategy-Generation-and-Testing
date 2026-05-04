#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and ADX regime filter
# Uses Camarilla pivot levels (R3, S3) from 1d for breakout entries, 1d volume spike for confirmation,
# and 1d ADX > 25 to filter for trending regimes. Designed for 20-50 trades/year (~80-200 total over 4 years)
# to minimize fee drag. Camarilla levels provide strong support/resistance, volume spike confirms
# institutional participation, and ADX ensures we only trade in trending markets to avoid whipsaw.
# Works in both bull/bear markets by adapting to trend direction via breakouts.

name = "4h_Camarilla_R3S3_1dVolumeSpike_ADXRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, volume, and ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R4 = Close + 1.5*(High-Low), R3 = Close + 1.125*(High-Low)
    #           S3 = Close - 1.125*(High-Low), S4 = Close - 1.5*(High-Low)
    hl_range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.125 * hl_range_1d
    camarilla_s3 = close_1d - 1.125 * hl_range_1d
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d volume spike (volume > 1.5 * 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate 1d ADX (14-period) for trend regime filter
    # ADX calculation: +DI, -DI, DX, then ADX = smoothed DX
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    # Shift arrays to align with original indices
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX and volume spike to 4h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(adx_14_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        trending_regime = adx_14_aligned[i] > 25
        
        if position == 0 and trending_regime:
            # Long conditions: price breaks above R3 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike_aligned[i] > 0.5):  # volume_spike is 0 or 1, use 0.5 threshold
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla range (between S3 and R3) OR ADX falls below 20
            if (close[i] >= camarilla_s3_aligned[i] and close[i] <= camarilla_r3_aligned[i]) or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Camarilla range OR ADX falls below 20
            if (close[i] >= camarilla_s3_aligned[i] and close[i] <= camarilla_r3_aligned[i]) or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals