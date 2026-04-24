#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band breakout with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for ADX trend strength and weekly pivot bias.
- Bollinger Bands (20, 2.0): Long when price breaks above upper band in strong uptrend (ADX>25),
  Short when price breaks below lower band in strong downtrend (ADX>25).
- Trend filter: Only trade when 1d ADX > 25 (strong trend) to avoid whipsaws in ranging markets.
- Weekly pivot bias: Use 1w Camarilla pivot levels - only take long if price above weekly H3,
  short if price below weekly L3 to align with higher timeframe structure.
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Uses Bollinger Bands for volatility-based breakout detection and ADX for trend strength filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original indices
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    tr_smoothed = WilderSmooth(tr, period)
    dm_plus_smoothed = WilderSmooth(dm_plus, period)
    dm_minus_smoothed = WilderSmooth(dm_minus, period)
    
    # Directional Indicators
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = WilderSmooth(dx, period)
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for weekly pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1d ADX for trend filter (strong trend > 25)
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1w Camarilla pivot levels for bias
    # Weekly pivot: (H+L+C)/3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    weekly_range = high_1w - low_1w
    # Camarilla levels: H3/L3 = pivot +/- 1.1 * range/2
    weekly_h3 = weekly_pivot + (1.1 * weekly_range / 2)
    weekly_l3 = weekly_pivot - (1.1 * weekly_range / 2)
    
    # Align weekly pivot levels to 6h
    weekly_h3_aligned = align_htf_to_ltf(prices, df_1w, weekly_h3)
    weekly_l3_aligned = align_htf_to_ltf(prices, df_1w, weekly_l3)
    
    # 6h Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_bb + (bb_std * std_bb)
    bb_lower = sma_bb - (bb_std * std_bb)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 20, 30)  # BB period + volume MA + ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(weekly_h3_aligned[i]) or np.isnan(weekly_l3_aligned[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(sma_bb[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in strong trend (ADX > 25)
            if adx_1d_aligned[i] > 25:
                # Long conditions: price breaks above upper BB AND above weekly H3
                if (close[i] > bb_upper[i]) and (close[i] > weekly_h3_aligned[i]) and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short conditions: price breaks below lower BB AND below weekly L3
                elif (close[i] < bb_lower[i]) and (close[i] < weekly_l3_aligned[i]) and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price falls below middle BB or weekly L3
            if close[i] < sma_bb[i] or close[i] < weekly_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above middle BB or weekly H3
            if close[i] > sma_bb[i] or close[i] > weekly_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerBreakout_1dADX_1wPivotBias_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0