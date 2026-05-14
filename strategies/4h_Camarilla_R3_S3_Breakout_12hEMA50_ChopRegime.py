#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hEMA50_ChopRegime
Hypothesis: Camarilla R3/S3 breakouts with volume spike and 12h EMA50 trend filter, plus choppiness regime to avoid ranging markets. 
R3/S3 are stronger reversal levels reducing false breakouts. In trending markets (CHOP < 38.2), breakouts with trend continuation yield profits.
In ranging markets (CHOP > 61.8), we avoid entries to reduce whipsaws. Volume spike confirms institutional participation.
Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    uptrend_12h = close > ema_50_12h_aligned
    downtrend_12h = close < ema_50_12h_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index regime filter (14-period)
    # CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.maximum(np.abs(low - np.roll(close, 1)), tr1)
    tr = np.where(np.arange(len(tr1)) == 0, high[0] - low[0], np.maximum(tr1, tr2))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * np.sqrt(14) / (highest_high14 - lowest_low14)) / np.log10(14)
    chop_regime = chop < 38.2  # Only trade in trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA + 50 for EMA + 14 for CHOP)
    start_idx = 80
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: close breaks above R3, with 12h uptrend, volume spike, and trending regime
            if (close[i] > camarilla_r3_aligned[i] and uptrend_12h[i] and 
                volume_spike[i] and chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S3, with 12h downtrend, volume spike, and trending regime
            elif (close[i] < camarilla_s3_aligned[i] and downtrend_12h[i] and 
                  volume_spike[i] and chop_regime[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close drops below S3 (mean reversion) OR 12h trend changes to downtrend OR chop regime becomes ranging
            if (close[i] < camarilla_s3_aligned[i] or not uptrend_12h[i] or not chop_regime[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close rises above R3 (mean reversion) OR 12h trend changes to uptrend OR chop regime becomes ranging
            if (close[i] > camarilla_r3_aligned[i] or not downtrend_12h[i] or not chop_regime[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_ChopRegime"
timeframe = "4h"
leverage = 1.0