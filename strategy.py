#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams %R reversal with 1-day ADX trend filter and volume confirmation.
# Uses Williams %R(14) for mean-reversion entries (oversold/overbought) in direction of daily trend.
# Works in bull/bear markets by only taking long signals in uptrend and short signals in downtrend.
# Target: 15-25 trades/year per symbol to minimize fee drag and maximize edge.
name = "12h_WilliamsR_1dADX_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 14-period ADX for trend strength and direction
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Directional Movement
    up_move = np.diff(high_1d)
    down_move = -np.diff(low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        alpha = 1.0 / period
        smoothed = np.zeros_like(data)
        smoothed[period-1] = np.nanmean(data[:period])  # simple average for first value
        for i in range(period, len(data)):
            smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
        return smoothed
    
    tr14 = WilderSmooth(tr, 14)
    plus_dm14 = WilderSmooth(plus_dm, 14)
    minus_dm14 = WilderSmooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = WilderSmooth(dx, 14)
    
    # Trend direction from +DI/-DI crossover
    plus_di14_shift = np.concatenate([[np.nan], plus_di14[:-1]])
    minus_di14_shift = np.concatenate([[np.nan], minus_di14[:-1]])
    uptrend = plus_di14 > minus_di14
    downtrend = minus_di14 > plus_di14
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    # Williams %R(14) on 12h data for mean-reversion signals
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) != 0, williams_r, -50)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema > 0, volume / vol_ema, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(uptrend_aligned[i]) or 
            np.isnan(downtrend_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Require minimum trend strength (ADX > 20)
        strong_trend = adx_aligned[i] > 20
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) in uptrend with volume spike
            long_condition = (williams_r[i] < -80) and uptrend_aligned[i] and vol_spike[i] and strong_trend
            # Short entry: Williams %R overbought (> -20) in downtrend with volume spike
            short_condition = (williams_r[i] > -20) and downtrend_aligned[i] and vol_spike[i] and strong_trend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Williams %R returns to neutral (> -50) or trend weakens
            if (williams_r[i] > -50) or (adx_aligned[i] < 20) or (~uptrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Williams %R returns to neutral (< -50) or trend weakens
            if (williams_r[i] < -50) or (adx_aligned[i] < 20) or (~downtrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals