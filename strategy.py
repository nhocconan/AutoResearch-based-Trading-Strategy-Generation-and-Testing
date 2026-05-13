#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation. The Bollinger Band squeeze identifies low volatility periods preceding explosive moves. ADX > 25 on daily timeframe confirms the market is trending (not ranging), reducing false breakouts. Volume spike (>2.0x 20-bar average) confirms institutional participation. Designed for BTC/ETH robustness: squeeze breakouts work in both bull and bear markets by capturing volatility expansion after consolidation, while ADX filter avoids whipsaws in ranging markets. Targets 12-37 trades/year on 6h timeframe.

name = "6h_BBandSqueeze_Breakout_1dADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20, 2) on 6h data
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # normalized width
    
    # Bollinger Band squeeze: width < 20th percentile of last 50 bars
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20).values
    squeeze_condition = bb_width < bb_width_percentile
    
    # Breakout condition: price closes outside Bollinger Bands
    breakout_upper = close > bb_upper
    breakout_lower = close < bb_lower
    
    # Calculate 1d ADX(14) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_smooth = WilderSmoothing(tr, 14)
    plus_dm_smooth = WilderSmoothing(plus_dm, 14)
    minus_dm_smooth = WilderSmoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = WilderSmoothing(dx, 14)
    
    # Align ADX to 6h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(bb_width[i]) or 
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # ENTRY CONDITIONS:
            # 1. Bollinger Band squeeze (low volatility)
            # 2. Price breaks out of Bollinger Bands
            # 3. 1d ADX > 25 (trending market)
            # 4. Volume spike > 2.0x 20-bar average
            if (squeeze_condition[i] and 
                (breakout_upper[i] or breakout_lower[i]) and 
                adx_aligned[i] > 25 and 
                volume[i] > 2.0 * avg_volume[i]):
                
                # Determine breakout direction
                if breakout_upper[i]:
                    signals[i] = 0.25
                    position = 1
                else:  # breakout_lower[i]
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to middle Bollinger Band OR squeeze re-establishes
            if (close[i] <= bb_middle[i] or squeeze_condition[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to middle Bollinger Band OR squeeze re-establishes
            if (close[i] >= bb_middle[i] or squeeze_condition[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals