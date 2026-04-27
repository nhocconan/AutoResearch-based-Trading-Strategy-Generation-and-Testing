#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot-based strategy using 1d levels for breakout/mean-reversion
# - In trending markets: break above R3 or below S3 with volume continuation
# - In ranging markets: fade at R4/S4 with mean reversion to pivot
# - Uses 1w trend filter to determine regime (trending vs ranging)
# - Designed to work in both bull (buy breakouts) and bear (sell breakdowns)
# - Target: 20-40 trades/year to avoid excessive fees

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    camarilla_pp = np.full(len(df_1d), np.nan)
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Previous day's values
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Pivot point
        pp = (ph + pl + pc) / 3
        camarilla_pp[i] = pp
        
        # Range
        range_val = ph - pl
        
        # Camarilla levels
        camarilla_r3[i] = pp + (range_val * 1.1 / 2)
        camarilla_s3[i] = pp - (range_val * 1.1 / 2)
        camarilla_r4[i] = pp + (range_val * 1.1)
        camarilla_s4[i] = pp - (range_val * 1.1)
    
    # Align Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Get weekly data for trend filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on weekly data
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values
        def smooth_wilder(arr, period):
            n = len(arr)
            result = np.full(n, np.nan)
            if n < period:
                return result
            
            # First value is simple average
            result[period-1] = np.nansum(arr[1:period]) / period
            
            # Wilder smoothing
            for i in range(period, n):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
                else:
                    result[i] = np.nan
            return result
        
        atr = smooth_wilder(tr, period)
        plus_dm_smooth = smooth_wilder(plus_dm, period)
        minus_dm_smooth = smooth_wilder(minus_dm, period)
        
        # Directional Indicators
        plus_di = np.where(atr != 0, (plus_dm_smooth / atr) * 100, 0)
        minus_di = np.where(atr != 0, (minus_dm_smooth / atr) * 100, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = smooth_wilder(dx, period)
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume spike: current volume > 2 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = max(20, 30)
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_1w_aligned[i] > 25
        is_ranging = adx_1w_aligned[i] < 20
        
        if position == 0:
            if is_trending:
                # Trending market: breakout strategy
                # Long: break above R3 with volume
                if (close[i] > r3_aligned[i] and 
                    volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: break below S3 with volume
                elif (close[i] < s3_aligned[i] and 
                      volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging market: mean reversion at extremes
                # Long: touch S4 with rejection
                if (low[i] <= s4_aligned[i] and 
                    close[i] > s4_aligned[i]):
                    signals[i] = 0.20
                    position = 1
                # Short: touch R4 with rejection
                elif (high[i] >= r4_aligned[i] and 
                      close[i] < r4_aligned[i]):
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime: no trade
                signals[i] = 0.0
        elif position == 1:
            # Long exit conditions
            if is_trending:
                # Exit on trend weakening or S3 touch
                if (adx_1w_aligned[i] < 20 or 
                    low[i] <= s3_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif is_ranging:
                # Exit on reversion to pivot or opposite extreme
                if (close[i] >= pp_aligned[i] or 
                    high[i] >= r4_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short exit conditions
            if is_trending:
                # Exit on trend weakening or R3 touch
                if (adx_1w_aligned[i] < 20 or 
                    high[i] >= r3_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif is_ranging:
                # Exit on reversion to pivot or opposite extreme
                if (close[i] <= pp_aligned[i] or 
                    low[i] <= s4_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
            else:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_Pivot_Regime_ADX_Volume_v1"
timeframe = "6h"
leverage = 1.0