#!/usr/bin/env python3
"""
6h_MarketRegime_Adaptive_Momentum
Hypothesis: On 6h timeframe, use 1d RSI and ATR to detect market regime (trending vs ranging).
In trending regime (ADX > 25): breakout of 6h Donchian(10) with volume confirmation.
In ranging regime (ADX <= 25): mean reversion at 6h Bollinger Bands (20,2) with RSI(14) filter.
This adapts to both bull and bear markets by switching between trend following and mean reversion.
Target: 50-150 total trades over 4 years = 12-37/year. Size: 0.25.
"""

name = "6h_MarketRegime_Adaptive_Momentum"
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

    # Get 1d data for regime detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # 1d ADX for regime detection
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    def smooth_wilder(arr, period):
        result = np.zeros_like(arr)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_smooth = smooth_wilder(tr, 14)
    plus_dm_smooth = smooth_wilder(plus_dm, 14)
    minus_dm_smooth = smooth_wilder(minus_dm, 14)
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx_1d = smooth_wilder(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    # 6h Donchian(10) for breakout signals
    donchian_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low = pd.Series(low).rolling(window=10, min_periods=10).min().values

    # 6h Bollinger Bands(20,2) for mean reversion
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20

    # 6h RSI(14) for mean reversion filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    rsi[avg_gain == 0] = 0

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Regime detection: ADX > 25 = trending, ADX <= 25 = ranging
            if adx_1d_aligned[i] > 25:
                # TRENDING REGIME: Donchian breakout with volume
                if close[i] > donchian_high[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # RANGING REGIME: Mean reversion at Bollinger Bands with RSI filter
                if close[i] < bb_lower[i] and rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                elif close[i] > bb_upper[i] and rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Opposite signal or volatility expansion
            if adx_1d_aligned[i] > 25:
                # In trending regime: exit on Donchian reversal
                if close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In ranging regime: exit at mean or RSI normalization
                if close[i] > sma_20[i] or rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Opposite signal or volatility expansion
            if adx_1d_aligned[i] > 25:
                # In trending regime: exit on Donchian reversal
                if close[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In ranging regime: exit at mean or RSI normalization
                if close[i] < sma_20[i] or rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25

    return signals