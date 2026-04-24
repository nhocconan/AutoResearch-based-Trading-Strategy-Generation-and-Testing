#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h ADX Trend Filter and Volume Spike.
- Primary timeframe: 6h for Bollinger Bands and volume spike detection.
- HTF: 12h ADX (>25) to confirm trending market (avoid false breakouts in ranging markets).
- Volume: Current 6h volume > 1.8 * 20-period volume MA to confirm breakout strength.
- Entry: Long when price breaks above upper BB AND 12h ADX > 25 AND volume spike.
         Short when price breaks below lower BB AND 12h ADX > 25 AND volume spike.
- Exit: Opposite BB break or loss of ADX/volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Bollinger squeeze captures low volatility breakouts which work in both bull and bear markets when combined with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands on 6h (20, 2)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma + (bb_std * std)
    lower_bb = sma - (bb_std * std)
    
    # Get 12h data for ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX on 12h (14-period)
    period = 14
    df_12h_high = df_12h['high'].values
    df_12h_low = df_12h['low'].values
    df_12h_close = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(df_12h_high[1:] - df_12h_close[:-1])
    tr2 = np.abs(df_12h_low[1:] - df_12h_close[:-1])
    tr3 = np.abs(df_12h_high[1:] - df_12h_low[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # Directional Movement
    up_move = df_12h_high[1:] - df_12h_high[:-1]
    down_move = df_12h_low[:-1] - df_12h_low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            else:
                result[i] = np.nan
        return result
    
    tr_smoothed = WilderSmoothing(tr, period)
    plus_dm_smoothed = WilderSmoothing(plus_dm, period)
    minus_dm_smoothed = WilderSmoothing(minus_dm, period)
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smoothed / tr_smoothed)
    minus_di = 100 * (minus_dm_smoothed / tr_smoothed)
    
    # DX and ADX
    dx = np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = WilderSmoothing(dx, period)
    
    # Calculate 20-period volume MA on 12h
    df_12h_volume = df_12h['volume'].values
    vol_ma_12h = pd.Series(df_12h_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 6h volume > 1.8 * 20-period 12h volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 2*period+20)  # BB period + ADX smoothing + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike and ADX > 25
            if volume_spike[i] and adx_val > 25:
                # Bullish: price breaks above upper BB
                if curr_close > upper_bb[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below lower BB
                elif curr_close < lower_bb[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below middle BB OR loss of ADX/volume confirmation
            if curr_close < sma[i] or adx_val <= 25 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above middle BB OR loss of ADX/volume confirmation
            if curr_close > sma[i] or adx_val <= 25 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_ADX25_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0