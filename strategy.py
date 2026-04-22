#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ADX trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes explosive moves in both bull and bear markets.
# Breakout direction confirmed by 1d ADX > 25 (strong trend) and volume > 2x average.
# Works in bull markets (upward breakouts) and bear markets (downward breakouts).
# Uses discrete position sizing (0.25) to minimize fee churn.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) for trend strength filter
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_14_1d = adx
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # 6h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    
    # Bollinger Band Squeeze: bandwidth < 50th percentile of last 50 periods
    bandwidth = (upper - lower) / sma
    bandwidth_percentile = pd.Series(bandwidth).rolling(window=50, min_periods=10).quantile(0.5).values
    squeeze = bandwidth < bandwidth_percentile
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50)  # BB period and bandwidth lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i]) or np.isnan(bandwidth_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for squeeze breakout with trend confirmation
            if squeeze[i-1] and not squeeze[i]:  # Squeeze just released
                # Breakout upward: close > upper band
                if close[i] > upper[i] and adx_14_1d_aligned[i] > 25 and volume[i] > 2 * vol_avg_20[i]:
                    signals[i] = 0.25
                    position = 1
                # Breakout downward: close < lower band
                elif close[i] < lower[i] and adx_14_1d_aligned[i] > 25 and volume[i] > 2 * vol_avg_20[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit: price returns to middle band (mean reversion) or volatility expands significantly
            if position == 1:
                # Exit long: price touches or goes below middle band
                if close[i] <= sma[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price touches or goes above middle band
                if close[i] >= sma[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_1dADX25_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0