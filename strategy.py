#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter (EMA50) and 1w volume regime confirmation.
# Long when BB width < 20th percentile (squeeze) AND price breaks above upper band AND 1d price > EMA50 AND 1w volume > 1.5 * 20-period average volume.
# Short when BB width < 20th percentile (squeeze) AND price breaks below lower band AND 1d price < EMA50 AND 1w volume > 1.5 * 20-period average volume.
# Exit when price returns to middle band (20-period SMA).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing low-volatility breakouts in alignment with higher timeframe trend and volume expansion.
# Target: 60-100 total trades over 4 years (15-25/year) for 6h timeframe.

name = "6h_BollingerSqueezeBreakout_1dEMA50_1wVolumeRegime_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate Bollinger Bands (20, 2) on primary timeframe
    basis = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    dev = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2 * dev
    lower_band = basis - 2 * dev
    bb_width = (upper_band - lower_band) / basis  # Normalized width
    
    # Calculate 20th percentile of BB width for squeeze condition (using expanding window to avoid look-ahead)
    bb_width_percentile = np.zeros(n)
    for i in range(20, n):
        bb_width_percentile[i] = np.percentile(bb_width[20:i+1], 20)
    squeeze_condition = bb_width < bb_width_percentile
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1w volume regime (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    volume_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume_1w > (1.5 * vol_ma_20_1w)  # High volume regime
    volume_regime_aligned = align_htf_to_ltf(prices, df_1w, volume_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after BB warmup
        # Skip if any required data is NaN
        if (np.isnan(basis[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Squeeze breakout above upper band AND 1d uptrend AND high volume regime
            if (squeeze_condition[i] and 
                close[i] > upper_band[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_regime_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Squeeze breakout below lower band AND 1d downtrend AND high volume regime
            elif (squeeze_condition[i] and 
                  close[i] < lower_band[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_regime_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to middle band
            if close[i] <= basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to middle band
            if close[i] >= basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals