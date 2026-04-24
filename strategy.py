#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d ADX Regime + Volume Spike
- Primary timeframe: 6h for execution, HTF: 1d for ADX regime filter.
- Williams %R(14) identifies overbought/oversold conditions: Long when %R crosses above -80 from below, Short when %R crosses below -20 from above.
- ADX(14) from 1d determines market regime: Only trade mean reversion when ADX < 25 (range market), avoid trending markets.
- Volume confirmation: current volume > 1.5x 20-period volume MA to ensure participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying oversold bounces in range, in bear via selling overbought bounces in range.
- Avoids whipsaws in strong trends by requiring low ADX regime.
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original index
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value: simple average
                result[period-1] = np.nansum(data[:period]) / period
                # Subsequent values: Wilder smoothing
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]):
                        result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = WilderSmoothing(tr, period)
        plus_dm_smooth = WilderSmoothing(plus_dm, period)
        minus_dm_smooth = WilderSmoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # Directional Index
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # ADX: smoothed DX
        adx = WilderSmoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams %R(14) on 6h
    def williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr.values
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Williams %R needs 14, volume MA needs 20, ADX needs ~34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(wr[i]) or np.isnan(wr[i-1]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in low ADX regime (range market)
            if adx_1d_aligned[i] < 25:
                # Williams %R signals: Long when crosses above -80 from below, Short when crosses below -20 from above
                wr_cross_up_80 = (wr[i-1] <= -80) and (wr[i] > -80)
                wr_cross_down_20 = (wr[i-1] >= -20) and (wr[i] < -20)
                
                if wr_cross_up_80 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif wr_cross_down_20 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) or ADX rises above 30 (trending)
            if (wr[i] >= -20) or (adx_1d_aligned[i] > 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) or ADX rises above 30 (trending)
            if (wr[i] <= -80) or (adx_1d_aligned[i] > 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0