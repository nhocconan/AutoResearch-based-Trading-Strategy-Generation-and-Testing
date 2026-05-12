#!/usr/bin/env python3
# 4h_12h_1d_Trix_VolumeSpike_Regime
# Hypothesis: Combines TRIX momentum (12h) with volume spike and Choppiness regime filter on 4h.
# Goes long when TRIX turns positive (bullish momentum) with volume spike in low-chop (trending) market.
# Goes short when TRIX turns negative (bearish momentum) with volume spike in low-chop market.
# Uses 1d ADX to confirm trend strength (ADX > 25) to avoid whipsaws in ranging markets.
# Designed for low trade frequency (<100/year) to minimize fee drag and improve generalization.

name = "4h_12h_1d_Trix_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.8x 30-period average (to reduce trade frequency)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 1h data for Choppiness index (regime filter)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # True Range
    tr1 = high_1h[1:] - low_1h[1:]
    tr2 = np.abs(high_1h[1:] - close_1h[:-1])
    tr3 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align index
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = high_1h[1:] - high_1h[:-1]
    down_move = low_1h[:-1] - low_1h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed +DM, -DM, TR
    def smooth(val, period):
        smoothed = np.zeros_like(val)
        smoothed[:] = np.nan
        if len(val) >= period:
            # First value: simple average
            smoothed[period-1] = np.nansum(val[:period])
            # Wilder smoothing
            for i in range(period, len(val)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + val[i]
        return smoothed
    
    smoothed_plus_dm = smooth(plus_dm, 14)
    smoothed_minus_dm = smooth(minus_dm, 14)
    smoothed_tr = smooth(tr, 14)
    
    # +DI and -DI
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr
    
    # DX and ADX
    dx = np.zeros_like(close_1h)
    dx[:] = np.nan
    di_sum = plus_di + minus_di
    mask = di_sum > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / di_sum[mask]
    
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_trending = adx > 25  # trending market filter
    
    # 12h data for TRIX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # TRIX: EMA of EMA of EMA of close, then ROC
    ema1 = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Rate of change of triple EMA
    trix = np.zeros_like(ema3)
    trix[:] = np.nan
    trix[12:] = (ema3[12:] - ema3[:-12]) / ema3[:-12] * 100
    
    # Align indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, prices, volume_spike)  # already 4h
    adx_trending_aligned = align_htf_to_ltf(prices, df_1h, adx_trending)
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(adx_trending_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX turns positive (>0) + volume spike + trending market (ADX>25)
            if (trix_aligned[i] > 0 and 
                trix_aligned[i-1] <= 0 and  # crossed above zero
                volume_spike_aligned[i] and
                adx_trending_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX turns negative (<0) + volume spike + trending market
            elif (trix_aligned[i] < 0 and 
                  trix_aligned[i-1] >= 0 and  # crossed below zero
                  volume_spike_aligned[i] and
                  adx_trending_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns negative OR loss of trend
            if (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0) or \
               (not adx_trending_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns positive OR loss of trend
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0) or \
               (not adx_trending_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals