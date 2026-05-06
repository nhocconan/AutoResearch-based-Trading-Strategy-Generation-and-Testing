#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining Bollinger Band squeeze breakout with 12h volume confirmation and ADX trend filter
# - Uses Bollinger Bands width percentile to identify low volatility squeeze (BBW < 20th percentile)
# - Entry on breakout above upper band or below lower band with volume confirmation (>1.5x average)
# - Uses 12h ADX > 25 to filter for trending markets only
# - Designed to capture explosive moves after consolidation periods in both bull and bear markets
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "4h_BollingerSqueeze_Breakout_12hVolume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) for squeeze detection
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    squeeze_condition = bb_width_percentile < 20  # Bollinger Band width in lowest 20%
    
    # 12h volume confirmation (average volume)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_12h_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_confirmation = volume > (1.5 * vol_12h_avg_aligned)
    
    # 12h ADX for trend filter
    if len(df_12h) < 15:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / np.where(tr_14 == 0, 0.0001, tr_14)
    di_minus = 100 * dm_minus_14 / np.where(tr_14 == 0, 0.0001, tr_14)
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 0.0001, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    trend_filter = adx_aligned > 25  # Only trade in trending markets
    
    # Breakout conditions
    long_breakout = (close > bb_upper) & squeeze_condition & volume_confirmation & trend_filter
    short_breakout = (close < bb_lower) & squeeze_condition & volume_confirmation & trend_filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Exit conditions: close back inside Bollinger Bands
        if position == 1 and close[i] >= bb_middle[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] <= bb_middle[i]:
            signals[i] = 0.0
            position = 0
        # Entry conditions
        elif position == 0:
            if long_breakout[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout[i]:
                signals[i] = -0.25
                position = -1
        # Hold position
        if position == 1 and signals[i] == 0:
            signals[i] = 0.25
        elif position == -1 and signals[i] == 0:
            signals[i] = -0.25
    
    return signals