#!/usr/bin/env python3
"""
Hypothesis: 4h Bollinger Band Width Breakout with 1d Volume Spike and ADX Trend Filter.
Bollinger Band Width contraction precedes expansion; breakouts from low volatility
capture momentum. Volume surge confirms breakout strength. ADX > 25 ensures trending
market to avoid whipsaws. Designed for ~30-50 trades/year to minimize fee drag,
works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for BBands, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    bb_mid = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Width percentile (250-day lookback for stability)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=250, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume spike: volume / 20-period average
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / vol_ma_20
    
    # ADX (14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align all indicators to 4h
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(bb_width_percentile_aligned[i]) or np.isnan(vol_ratio_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        bb_width_pct = bb_width_percentile_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        adx_val = adx_aligned[i]
        bb_upper = bb_upper_aligned[i]
        bb_lower = bb_lower_aligned[i]
        
        # Entry conditions: low volatility (BB width < 20th percentile) + volume spike + trend
        if position == 0:
            if bb_width_pct < 20 and vol_ratio_val > 1.5 and adx_val > 25:
                # Breakout above upper band -> long
                if price_close > bb_upper:
                    signals[i] = 0.25
                    position = 1
                # Breakout below lower band -> short
                elif price_close < bb_lower:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit: volatility expansion (BB width > 80th percentile) or trend weakening
            if bb_width_pct > 80 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_BBW_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0