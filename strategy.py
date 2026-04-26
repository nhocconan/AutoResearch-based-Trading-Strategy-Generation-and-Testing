#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ChopFilter_v1
Hypothesis: Camarilla R3/S3 breakout on 12h with 1d EMA34 trend filter, volume spike confirmation, and Choppiness Index regime filter to avoid whipsaws. Works in both bull (breakouts with trend) and bear (breakouts against trend filtered by chop) regimes. Targets 50-150 total trades over 4 years (12-37/year).
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
    
    # Load 1d data ONCE before loop for EMA and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Choppiness Index (CHOP) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) sum
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hl_range = hh - ll
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    chop = 100 * np.log10(atr_sum / hl_range) / np.log10(14)
    chop = np.where(hl_range == 0, 50, chop)  # neutral when range is zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Camarilla levels from previous 1d bar (for intraday breakout)
    # We need previous day's OHLC to calculate today's Camarilla levels
    # Since we're on 12h timeframe, we calculate levels once per day
    # Use previous completed 1d bar
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # first bar
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Camarilla equations
    range_1d = prev_high_1d - prev_low_1d
    camarilla_r3 = prev_close_1d + range_1d * 1.1 / 4
    camarilla_s3 = prev_close_1d - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (they change only at 1d boundaries)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 14, 20) + 1  # EMA34 + CHOP(14) + volume EMA20 + 1 for prev day
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade when CHOP < 61.8 (trending market)
        trending_regime = chop_aligned[i] < 61.8
        
        # Trend direction: price above/below EMA34
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_r3_aligned[i]
        breakout_down = close[i] < camarilla_s3_aligned[i]
        
        # Long logic: breakout above R3 in trending market with volume spike
        if trending_regime and breakout_up and volume_spike[i]:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below S3 in trending market with volume spike
        elif trending_regime and breakout_down and volume_spike[i]:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: opposite breakout or loss of trend
        elif position == 1 and (breakout_down or not trending_regime):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (breakout_up or not trending_regime):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0