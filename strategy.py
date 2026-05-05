#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Camarilla pivot levels (R3/S3) from 1d timeframe with volume spike and ADX trend filter
# Long when price breaks above R3 level AND ADX > 25 (trending) AND volume > 2.0 * avg_volume(20)
# Short when price breaks below S3 level AND ADX > 25 (trending) AND volume > 2.0 * avg_volume(20)
# Exit when price returns to pivot point (PP) level OR ADX drops below 20 (range market)
# Uses discrete sizing 0.30 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels provide statistically significant support/resistance
# Volume spike confirms breakout strength and reduces false signals
# ADX filter ensures we only trade in trending markets, avoiding whipsaws in ranging conditions
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "4h_Camarilla_R3S3_Breakout_1dADX25_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for Camarilla levels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d timeframe
    # PP = (High + Low + Close) / 3
    # R3 = PP + (High - Low) * 1.1/2
    # S3 = PP - (High - Low) * 1.1/2
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2.0
    s3_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (wait for completed daily bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate ADX(14) on 1d timeframe for trend filter
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / np.where(atr != 0, atr, 1e-10)
    di_minus = 100 * dm_minus_smooth / np.where(atr != 0, atr, 1e-10)
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) != 0, (di_plus + di_minus), 1e-10)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 AND ADX > 25 (trending) AND volume confirmation
            if (close[i] > r3_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below S3 AND ADX > 25 (trending) AND volume confirmation
            elif (close[i] < s3_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price returns to pivot point OR ADX drops below 20 (range market)
            if close[i] <= pp_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price returns to pivot point OR ADX drops below 20 (range market)
            if close[i] >= pp_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals