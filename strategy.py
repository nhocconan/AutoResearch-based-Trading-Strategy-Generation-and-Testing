#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 12h ADX Regime + Volume Spike
- Primary timeframe: 6h for execution, HTF: 12h for ADX regime filter.
- Williams %R(14) from 6h data: Long when %R crosses above -80 from below (oversold bounce),
  Short when %R crosses below -20 from above (overbought rejection).
- Regime filter: Only take longs when 12h ADX(14) < 25 (range/weak trend) and shorts when ADX > 25 (strong trend).
  This adapts to market conditions: mean reversion in chop, trend following in trends.
- Volume confirmation: current 6h volume > 1.8x 20-period volume MA to ensure participation.
- Discrete signal size: 0.25 to balance opportunity and risk.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying oversold bounces in weak trends, in bear via selling overbought rejection in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R(14) on 6h data
    def calculate_williams_r(high, low, close, window=14):
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr = calculate_williams_r(high, low, close, 14)
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h ADX(14) for regime filter
    def calculate_adx(high, low, close, window=14):
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr[0] = high[0] - low[0]  # first TR
        
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        plus_di = 100 * pd.Series(plus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=window, adjust=False, min_periods=window).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Williams %R needs 14, but we use 34 for safety with EMA-like smoothing in ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(wr[i]) or np.isnan(wr[i-1]) if i > 0 else True or
            np.isnan(adx_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            wr_prev = wr[i-1] if i > 0 else wr[i]
            wr_curr = wr[i]
            
            # Long: Williams %R crosses above -80 from below (oversold bounce)
            # Only in weak trend/range (ADX < 25)
            if wr_prev <= -80 and wr_curr > -80 and adx_12h_aligned[i] < 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above (overbought rejection)
            # Only in strong trend (ADX > 25)
            elif wr_prev >= -20 and wr_curr < -20 and adx_12h_aligned[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) or reverse signal
            if wr[i] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) or reverse signal
            if wr[i] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0