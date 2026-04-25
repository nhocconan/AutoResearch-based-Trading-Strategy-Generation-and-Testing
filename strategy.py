#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_ChopRegime_1dTrend
Hypothesis: 12h TRIX (triple-smoothed EMA) momentum with 1d trend filter (price >/< EMA34), volume confirmation (>2.0x 20-bar avg), and choppiness regime filter (CHOP > 61.8 = ranging, mean-reversion; CHOP < 38.2 = trending). 
Enters long when TRIX crosses above zero in 1d uptrend with volume spike and chop < 38.2 (trending market). 
Enters short when TRIX crosses below zero in 1d downtrend with volume spike and chop < 38.2. 
Exits on opposite TRIX cross or trend reversal. 
Designed for 12h timeframe with ~12-25 trades/year, works in bull/bear by following 1d trend filter and momentum confirmation with regime awareness.
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
    
    # 1d data for HTF trend filter and chop regime
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR for chop regime (14-period)
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # 1d Choppiness Index (CHOP) = 100 * log10(sum(atr14) / (max(high)-min(low))) / log10(14)
    sum_atr14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid div by zero
    chop_raw = 100 * np.log10(sum_atr14 / chop_denominator) / np.log10(14)
    chop_1d = np.where(np.isnan(chop_raw), 50.0, chop_raw)  # fill NaN with neutral
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # TRIX calculation on 12h timeframe (triple EMA 12-period)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100  # percentage change
    trix_values = trix.values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough warmup for TRIX (3*12=36) and 1d indicators
    start_idx = max(36, 34, 14, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(trix_values[i]) or 
            np.isnan(trix_values[i-1]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero (bullish momentum) in 1d uptrend with volume spike and chop < 38.2 (trending)
            trix_cross_up = (trix_values[i-1] <= 0) and (trix_values[i] > 0)
            long_setup = trix_cross_up and (close[i] > ema_34_1d_aligned[i]) and volume_spike[i] and (chop_1d_aligned[i] < 38.2)
            # Short: TRIX crosses below zero (bearish momentum) in 1d downtrend with volume spike and chop < 38.2
            trix_cross_down = (trix_values[i-1] >= 0) and (trix_values[i] < 0)
            short_setup = trix_cross_down and (close[i] < ema_34_1d_aligned[i]) and volume_spike[i] and (chop_1d_aligned[i] < 38.2)
            
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
            # Exit: TRIX crosses below zero OR trend turns down
            trix_cross_down = (trix_values[i-1] >= 0) and (trix_values[i] < 0)
            if trix_cross_down or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR trend turns up
            trix_cross_up = (trix_values[i-1] <= 0) and (trix_values[i] > 0)
            if trix_cross_up or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_TRIX_VolumeSpike_ChopRegime_1dTrend"
timeframe = "12h"
leverage = 1.0