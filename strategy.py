#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme + 1d ADX Regime + Volume Spike.
- Primary timeframe: 4h for execution, HTF: 1d for ADX regime filter.
- Williams %R(14): Long when < -80 (oversold), Short when > -20 (overbought).
- Regime filter: ADX(14) > 25 = trending (trade with trend), ADX < 20 = ranging (mean revert at extremes).
- Volume confirmation: current volume > 2.0x 20-period volume MA to filter low-quality signals.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying dips in uptrend, in bear via selling rallies in downtrend, and mean reverting in ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R(14) for overbought/oversold
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Calculate 1d ADX(14) for regime filter
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_di_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    plus_di = 100 * plus_di_smooth / atr
    minus_di = 100 * minus_di_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe (completed 1d bar only)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1  # Need Williams %R(14)+buffer, volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime: ADX > 25 = trending, ADX < 20 = ranging
            if adx_aligned[i] > 25:
                # Trending regime: trade pullbacks in trend direction
                # Need EMA50 slope for trend direction - calculate from 1d close
                ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
                ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
                if not np.isnan(ema_50_1d_aligned[i]) and not np.isnan(ema_50_1d_aligned[i-1]):
                    ema50_slope = ema_50_1d_aligned[i] - ema_50_1d_aligned[i-1]
                    if ema50_slope > 0 and williams_r_aligned[i] < -80 and volume_spike[i]:
                        # Uptrend: buy on oversold
                        signals[i] = 0.25
                        position = 1
                    elif ema50_slope < 0 and williams_r_aligned[i] > -20 and volume_spike[i]:
                        # Downtrend: sell on overbought
                        signals[i] = -0.25
                        position = -1
            elif adx_aligned[i] < 20:
                # Ranging regime: mean revert at extremes
                if williams_r_aligned[i] < -80 and volume_spike[i]:
                    # Oversold: buy
                    signals[i] = 0.25
                    position = 1
                elif williams_r_aligned[i] > -20 and volume_spike[i]:
                    # Overbought: sell
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns to neutral or overbought
            if williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to neutral or oversold
            if williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0