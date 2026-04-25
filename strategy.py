#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA200_TrendFilter_VolumeSpike_WithRegime
Hypothesis: Trade Camarilla R3/S3 breakouts with 1d EMA200 trend filter and volume spike confirmation, 
adding a daily choppiness regime filter to avoid ranging markets. Only trade when CHOP(1d) > 61.8 (trending regime).
EMA200 provides robust long-term trend filter reducing whipsaws in bear markets. 
R3/S3 are stronger levels reducing false breakouts. Discrete sizing 0.25 to manage risk and minimize fee churn.
Target: 20-40 trades/year to stay within fee drag limits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and regime
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA200 for trend filter (more robust than EMA50)
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate daily choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * (HHV - LLV))) / log10(n)
    # We'll use a simplified version: CHOP = 100 * log10(ATR(1) sum over 14 days / (14 * (max(high)-min(low)) over 14 days)) / log10(14)
    tr1 = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # first TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = 14 * (hh14 - ll14)
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid div by zero
    chop_raw = np.sum(pd.Series(atr1).rolling(window=14, min_periods=14).sum().values.reshape(-1, 14), axis=1) / chop_denominator
    # Fix shape: we need to compute properly
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / chop_denominator) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # default to middle if NaN
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_day_high - prev_day_low
    r3 = prev_day_close + 1.1 * camarilla_range / 4  # R3 level
    s3 = prev_day_close - 1.1 * camarilla_range / 4  # S3 level
    h3 = prev_day_close + 1.1 * camarilla_range / 6  # H3 level
    l3 = prev_day_close - 1.1 * camarilla_range / 6  # L3 level
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: current volume > 2.0x 20-period average (stricter filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for daily EMA200 (200), chop (14), and volume MA (20)
    start_idx = max(200, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP > 61.8)
        in_trending_regime = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above R3 AND daily trend bullish (close > EMA200) AND volume spike AND trending regime
            long_setup = (close[i] > r3_aligned[i]) and \
                         (close[i] > ema_200_1d_aligned[i]) and \
                         volume_spike[i] and \
                         in_trending_regime
            # Short: price breaks below S3 AND daily trend bearish (close < EMA200) AND volume spike AND trending regime
            short_setup = (close[i] < s3_aligned[i]) and \
                          (close[i] < ema_200_1d_aligned[i]) and \
                          volume_spike[i] and \
                          in_trending_regime
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Camarilla H3/L3 range OR daily trend turns bearish OR regime changes to ranging
            if ((close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
                (close[i] < ema_200_1d_aligned[i]) or \
                (chop_aligned[i] <= 61.8)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H3/L3 range OR daily trend turns bullish OR regime changes to ranging
            if ((close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
                (close[i] > ema_200_1d_aligned[i]) or \
                (chop_aligned[i] <= 61.8)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA200_TrendFilter_VolumeSpike_WithRegime"
timeframe = "4h"
leverage = 1.0