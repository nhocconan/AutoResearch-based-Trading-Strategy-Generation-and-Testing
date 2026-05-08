#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation and 12h ADX trend filter.
# Designed for low trade frequency (12-37 trades/year) to avoid fee drag. Uses daily Camarilla levels (S3/S2/S1/R1/R2/R3)
# as support/resistance structure, volume surge for momentum confirmation, and ADX to filter ranging markets.
# Works in bull/bear markets by trading breakouts in trending regimes and mean-reverting at extremes in ranging regimes.

name = "12h_Camarilla_S3R3_VolumeADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 6)
    # S2 = C - (Range * 1.1 / 2)
    # S3 = C - (Range * 1.1)
    # R1 = C + (Range * 1.1 / 6)
    # R2 = C + (Range * 1.1 / 2)
    # R3 = C + (Range * 1.1)
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    s1 = close_1d - (range_1d * 1.1 / 6)
    s2 = close_1d - (range_1d * 1.1 / 2)
    s3 = close_1d - (range_1d * 1.1)
    r1 = close_1d + (range_1d * 1.1 / 6)
    r2 = close_1d + (range_1d * 1.1 / 2)
    r3 = close_1d + (range_1d * 1.1)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Get 12h data for volume and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume spike: 2x 20-period EMA
    vol_ema = pd.Series(volume_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume_12h > (vol_ema * 2.0)
    
    # ADX (14-period) for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        atr = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        # Initial values (simple average)
        if len(tr) >= period + 1:
            atr[period] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
            
            # Wilder's smoothing
            for i in range(period + 1, len(tr)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
        
        # DI+ and DI-
        di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        
        # DX
        dx = np.where((di_plus + di_minus) != 0, 
                      100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        
        # ADX (smoothed DX)
        adx = np.full_like(dx, np.nan)
        if len(dx) >= 2 * period:
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            for i in range(2*period, len(dx)):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Align volume spike and ADX to 12h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike and ADX > 20 (trending)
            if close[i] > r3_aligned[i] and vol_spike_aligned[i] and adx_aligned[i] > 20:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume spike and ADX > 20 (trending)
            elif close[i] < s3_aligned[i] and vol_spike_aligned[i] and adx_aligned[i] > 20:
                signals[i] = -0.25
                position = -1
            # Mean reversion in ranging markets (ADX <= 20): buy at S3, sell at R3
            elif adx_aligned[i] <= 20:
                if close[i] <= s3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= r3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or ADX drops below 15 (trend weakening)
            if close[i] < s3_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or ADX drops below 15 (trend weakening)
            if close[i] > r3_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals