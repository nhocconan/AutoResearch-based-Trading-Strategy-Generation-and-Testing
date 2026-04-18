#!/usr/bin/env python3
"""
4h_MarketRegime_ADX_Trend_With_Volume_Confirmation
Hypothesis: In trending markets (ADX > 25), enter long on pullbacks to EMA21 with volume confirmation;
             in ranging markets (ADX < 20), fade extremes at Bollinger Bands with volume divergence.
             Uses 1d ADX for regime classification to avoid whipsaw. Target: 25-40 trades/year.
"""

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
    
    # 1d ADX for regime classification
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    def _wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # first value: simple average
        result[period-1] = np.nansum(arr[1:period])  # skip index 0
        # subsequent: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = _wilder_smooth(tr, 14)
    plus_di = 100 * _wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * _wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = _wilder_smooth(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # EMA21 on 4h for pullback entries
    close_s = pd.Series(close)
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Bollinger Bands (20, 2)
    sma20 = close_s.rolling(window=20, min_periods=20).mean().values
    std20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_conf = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 21)  # Warmup for ADX and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or
            np.isnan(ema21[i]) or
            np.isnan(upper_bb[i]) or
            np.isnan(lower_bb[i]) or
            np.isnan(volume_conf[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        price = close[i]
        ema21_val = ema21[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # Trending regime: ADX > 25
            if adx_val > 25:
                # Long: pullback to EMA21 in uptrend (price > EMA21 previously)
                if price > ema21_val and close[i-1] > ema21[i-1] and vol_conf:
                    signals[i] = 0.25
                    position = 1
                # Short: pullback to EMA21 in downtrend (price < EMA21 previously)
                elif price < ema21_val and close[i-1] < ema21[i-1] and vol_conf:
                    signals[i] = -0.25
                    position = -1
            # Ranging regime: ADX < 20
            elif adx_val < 20:
                # Long: price at lower BB with volume confirmation
                if price <= lower_bb[i] and vol_conf:
                    signals[i] = 0.25
                    position = 1
                # Short: price at upper BB with volume confirmation
                elif price >= upper_bb[i] and vol_conf:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: ADX drops (trend ending) or opposite BB touch
            if adx_val < 20 and price >= upper_bb[i]:
                signals[i] = 0.0
                position = 0
            elif adx_val > 25 and price < ema21_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: ADX drops (trend ending) or opposite BB touch
            if adx_val < 20 and price <= lower_bb[i]:
                signals[i] = 0.0
                position = 0
            elif adx_val > 25 and price > ema21_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_MarketRegime_ADX_Trend_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0