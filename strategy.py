#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ADX trend filter and volume confirmation. Uses 1d Camarilla R4/S4 levels for breakout confirmation and profit targets. Designed for low-volatility breakouts in both bull and bear markets: Bollinger Band width < 20th percentile identifies squeeze, ADX > 25 confirms trend readiness, volume > 2x 20-bar average validates breakout, and Camarilla R4/S4 levels provide structured entries and exits. Targets 12-37 trades/year on 6h timeframe.

name = "6h_BBandSqueeze_Breakout_1dADX_VolumeConfirm_CamarillaExits_v1"
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
    
    # Calculate 20-period Bollinger Bands on 6h data
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = (bb_upper - bb_lower) / sma_20
    
    # Calculate Bollinger Band width percentile (lookback 100 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=100, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate 1d ADX for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(high_1d) - pd.Series(close_1d).shift(1)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(low_1d)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and ADX
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Calculate 1d Camarilla levels for breakout confirmation and exits
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after lookback for percentile
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bollinger Band squeeze (width < 20th percentile), ADX > 25, volume spike (>2x avg), close above Camarilla R4
            if (bb_width_percentile[i] < 20 and 
                adx_1d_aligned[i] > 25 and 
                volume[i] > 2.0 * avg_volume[i] and 
                close[i] > camarilla_r4_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bollinger Band squeeze (width < 20th percentile), ADX > 25, volume spike (>2x avg), close below Camarilla S4
            elif (bb_width_percentile[i] < 20 and 
                  adx_1d_aligned[i] > 25 and 
                  volume[i] > 2.0 * avg_volume[i] and 
                  close[i] < camarilla_s4_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches Camarilla R4 again (mean reversion) OR Bollinger Band width expands (> 80th percentile)
            if (close[i] <= camarilla_r4_aligned[i] or 
                bb_width_percentile[i] > 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches Camarilla S4 again (mean reversion) OR Bollinger Band width expands (> 80th percentile)
            if (close[i] >= camarilla_s4_aligned[i] or 
                bb_width_percentile[i] > 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals