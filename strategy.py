#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX trend filter and volume confirmation
# Williams %R(14) identifies overbought/oversold conditions on 6h chart
# 1d ADX > 25 ensures we only trade in trending markets to avoid whipsaws in ranging conditions
# Volume confirmation (>1.3x 20 EMA volume) filters low-momentum breakouts
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "6h_WilliamsR_1dADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) with proper smoothing
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        def wilders_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = wilders_smooth(tr, period)
        plus_dm_smooth = wilders_smooth(plus_dm, period)
        minus_dm_smooth = wilders_smooth(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, (plus_dm_smooth / atr) * 100, 0)
        minus_di = np.where(atr != 0, (minus_dm_smooth / atr) * 100, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 
                      np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = wilders_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_shifted = np.roll(adx_1d, 1)  # Use prior completed 1d bar
    adx_1d_shifted[0] = np.nan
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_shifted)
    
    # Calculate Williams %R(14) on 6h
    def williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        # Williams %R = -100 * (HH - Close) / (HH - LL)
        wr = np.full_like(close, np.nan)
        mask = (highest_high != lowest_low) & ~np.isnan(highest_high) & ~np.isnan(lowest_low)
        wr[mask] = -100 * (highest_high[mask] - close[mask]) / (highest_high[mask] - lowest_low[mask])
        return wr
    
    wr_6h = williams_r(high, low, close, 14)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(wr_6h[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) AND ADX > 25 (trending) AND volume spike
            if wr_6h[i] < -80 and adx_1d_aligned[i] > 25 and volume[i] > (1.3 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought (> -20) AND ADX > 25 (trending) AND volume spike
            elif wr_6h[i] > -20 and adx_1d_aligned[i] > 25 and volume[i] > (1.3 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 OR ADX weakens (< 20)
            if wr_6h[i] > -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 OR ADX weakens (< 20)
            if wr_6h[i] < -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals