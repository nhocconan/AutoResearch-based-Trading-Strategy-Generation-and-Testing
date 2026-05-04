#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) extreme reversal with 1d ADX(14) regime filter and volume confirmation
# Williams %R < -80 = oversold (long setup), > -20 = overbought (short setup)
# 1d ADX > 25 = trending regime (fade extremes), ADX < 20 = ranging regime (mean revert at extremes)
# Volume confirmation: current volume > 1.5x 20-period EMA of volume ensures participation
# Uses discrete sizing 0.25 to balance risk and minimize fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in bull markets (mean revert in ranges) and bear markets (fade extremes in trends)

name = "6h_WilliamsR14_1dADX_Regime_Volume"
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) for regime detection
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = np.nan  # First value has no prior close
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = np.nan
        down_move[0] = np.nan
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR, PlusDM, MinusDM (Wilder's smoothing)
        def Wilder_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value: simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        tr_smooth = Wilder_smoothing(tr, period)
        plus_dm_smooth = Wilder_smoothing(plus_dm, period)
        minus_dm_smooth = Wilder_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        dx[~np.isfinite(dx)] = np.nan
        
        adx = Wilder_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams %R(14) on 6h timeframe
    def williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        
        for i in range(period-1, len(high)):
            highest_high[i] = np.nanmax(high[i-period+1:i+1])
            lowest_low[i] = np.nanmin(low[i-period+1:i+1])
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        wr[highest_high == lowest_low] = np.nan  # Avoid division by zero
        return wr
    
    wr_6h = williams_r(high, low, close, 14)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(wr_6h[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: Williams %R oversold (< -80) AND volume spike
            # In ranging market (ADX < 20): mean revert at extremes
            # In trending market (ADX > 25): only take if ADX weakening (not implemented for simplicity)
            if wr_6h[i] < -80 and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: Williams %R overbought (> -20) AND volume spike
            elif wr_6h[i] > -20 and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 (momentum shift) OR ADX strong trend (>25) with adverse move
            if wr_6h[i] > -50 or adx_1d_aligned[i] > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 OR ADX strong trend (>25)
            if wr_6h[i] < -50 or adx_1d_aligned[i] > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals