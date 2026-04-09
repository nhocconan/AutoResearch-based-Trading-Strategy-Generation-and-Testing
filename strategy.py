#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h EMA crossover with 1d ADX trend filter and volume spike confirmation
# EMA crossover (21/55) on 12h captures medium-term trend direction
# 1d ADX > 25 ensures we only trade in trending markets (avoids chop/range)
# Volume spike (current 6h volume > 2.0x 20-period average) confirms breakout momentum
# Position size fixed at 0.25 to balance risk and avoid fee churn
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# Works in both bull and bear by only taking trades aligned with strong trend (ADX filter)

name = "6h_12h_ema_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA crossover (21/55)
    ema_21 = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_55 = pd.Series(close_12h).ewm(span=55, min_periods=55, adjust=False).mean().values
    ema_cross = ema_21 - ema_55  # Positive = bullish crossover, Negative = bearish
    
    # Load 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    plus_di_1d = 100 * wilder_smooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilder_smooth(dx_1d, 14)
    
    # Align indicators to 6h timeframe
    ema_cross_aligned = align_htf_to_ltf(prices, df_12h, ema_cross)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_cross_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i]) or adx_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x average 6h volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        trend_filter = adx_aligned[i] > 25
        
        if not (volume_confirmed and trend_filter):
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when EMA crossover turns bearish
            if ema_cross_aligned[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when EMA crossover turns bullish
            if ema_cross_aligned[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Enter long on bullish EMA crossover with volume and trend confirmation
            # Enter short on bearish EMA crossover with volume and trend confirmation
            if volume_confirmed and trend_filter:
                if ema_cross_aligned[i] > 0:  # Bullish crossover
                    position = 1
                    signals[i] = position_size
                elif ema_cross_aligned[i] < 0:  # Bearish crossover
                    position = -1
                    signals[i] = -position_size
    
    return signals