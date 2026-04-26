#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Regime_v1
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume confirmation (1.8x), and choppiness regime filter (CHOP > 61.8 for ranging markets). 
Optimized for 12h timeframe to target 50-150 trades over 4 years by balancing signal frequency and filter tightness. 
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear via 1d trend alignment and regime filter to avoid whipsaws in strong trends.
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
    
    # Load 1d data ONCE before loop for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Calculate Camarilla pivot levels from 1d OHLC (using previous day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: volume > 1.8 * volume_ma(20) for balanced confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    # Choppiness Index regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    # We use CHOP > 61.8 to filter for ranging markets where Camarilla mean reversion works best
    tr_range = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    atr_14 = pd.Series(close).rolling(window=14, min_periods=14).apply(
        lambda x: np.mean(np.abs(np.diff(x))), raw=True
    ).values
    chop = 100 * np.log10(tr_range / (atr_14 * 14)) / np.log10(10)
    chop_regime = chop > 61.8  # ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA, 14 for CHOP)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(trend_1d[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R3/S3 breakout conditions with volume, trend, and regime confirmation
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND 1d uptrend AND volume spike AND ranging regime
            if close[i] > camarilla_r3_aligned[i] and trend_1d[i] == 1 and volume_spike[i] and chop_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND 1d downtrend AND volume spike AND ranging regime
            elif close[i] < camarilla_s3_aligned[i] and trend_1d[i] == -1 and volume_spike[i] and chop_regime[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR 1d trend turns down OR regime changes to trending
            if close[i] < camarilla_s3_aligned[i] or trend_1d[i] == -1 or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR 1d trend turns up OR regime changes to trending
            if close[i] > camarilla_r3_aligned[i] or trend_1d[i] == 1 or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Regime_v1"
timeframe = "12h"
leverage = 1.0