#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power + 1d ADX Regime Filter + Volume Confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for ADX regime and Elder Ray calculation.
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close).
- Regime filter: ADX(14) > 25 = trending (only trade in direction of EMA50 slope), ADX < 20 = ranging (mean revert at Elder Ray extremes).
- Volume confirmation: current volume > 1.5x 20-period volume MA to avoid low-volatility false signals.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying strong Bull Power in uptrend, in bear via selling strong Bear Power in downtrend, and mean reverting in ranges.
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
    
    # Calculate 1d EMA13 for Elder Ray (using close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d EMA50 for trend slope
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ADX(14) for regime filter
    # ADX requires +DI, -DI, and DX calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_di_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    plus_di = 100 * plus_di_smooth / atr
    minus_di = 100 * minus_di_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Elder Ray: Bull/Bear Power
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align all 1d indicators to 6h timeframe (completed 1d bar only)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1  # Need EMA50, volume MA(20), plus buffer
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        ema50_slope = ema_50_1d_aligned[i] - ema_50_1d_aligned[i-1]
        
        if position == 0:
            # Regime: ADX > 25 = trending, ADX < 20 = ranging
            if adx_aligned[i] > 25:
                # Trending regime: trade in direction of EMA50 slope
                if ema50_slope > 0 and bull_power_aligned[i] > 0 and volume_spike[i]:
                    # Uptrend: buy on bullish power
                    signals[i] = 0.25
                    position = 1
                elif ema50_slope < 0 and bear_power_aligned[i] < 0 and volume_spike[i]:
                    # Downtrend: sell on bearish power
                    signals[i] = -0.25
                    position = -1
            elif adx_aligned[i] < 20:
                # Ranging regime: mean revert at Elder Ray extremes
                if bull_power_aligned[i] < -0.5 * np.std(bull_power_aligned[max(0,i-50):i]) and volume_spike[i]:
                    # Oversold: buy
                    signals[i] = 0.25
                    position = 1
                elif bear_power_aligned[i] > 0.5 * np.std(bear_power_aligned[max(0,i-50):i]) and volume_spike[i]:
                    # Overbought: sell
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Elder Ray turns negative or trend reversal
            if bull_power_aligned[i] < 0 or ema50_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Elder Ray turns positive or trend reversal
            if bear_power_aligned[i] > 0 or ema50_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0