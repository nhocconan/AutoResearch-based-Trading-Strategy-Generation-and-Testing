#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power + 1d Regime Filter (ADX + EMA200) for BTC/ETH
- Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
- Measures bull/bear strength relative to short-term trend (EMA13)
- Regime filter: 1d ADX > 25 = trending (trade in direction of 1d EMA200)
- 1d ADX <= 25 = ranging (fade extremes using Bollinger Bands)
- Volume confirmation: > 1.5x 20-period average
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by adapting to regime
- Elder Ray provides clear momentum signals, regime filter avoids wrong-context trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for ADX and EMA
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(200) for long-term trend
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1d ADX(14) for regime detection
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilder_smooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h indicators
    # EMA13 for Elder Ray power calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = high - ema_13  # High minus EMA
    bear_power = ema_13 - low   # EMA minus Low
    
    # Bollinger Bands(20,2) for ranging regime
    bb_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 20)  # EMA200 and BB need most data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(ema_13[i]) or np.isnan(bb_ma[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine regime: 1d ADX > 25 = trending, <= 25 = ranging
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] <= 25
        
        # Volume filter
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            if is_trending and volume_ok:
                # Trending regime: trade with 1d EMA200 trend
                uptrend = close[i] > ema_200_aligned[i]
                downtrend = close[i] < ema_200_aligned[i]
                
                # Long: strong bull power in uptrend
                if bull_power[i] > 0 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: strong bear power in downtrend
                elif bear_power[i] > 0 and downtrend:
                    signals[i] = -0.25
                    position = -1
                    
            elif is_ranging and volume_ok:
                # Ranging regime: fade extremes at Bollinger Bands
                # Long: price at lower band with bullish momentum
                if close[i] <= bb_lower[i] and bull_power[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Short: price at upper band with bearish momentum
                elif close[i] >= bb_upper[i] and bear_power[i] > 0:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: bear power turns positive (momentum shift) or opposite BB touch
                if bear_power[i] > 0 or close[i] >= bb_upper[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: bull power turns positive or opposite BB touch
                if bull_power[i] > 0 or close[i] <= bb_lower[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_Power_1dADX_Regime_BB_VolumeConfirm"
timeframe = "6h"
leverage = 1.0