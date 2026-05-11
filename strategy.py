#!/usr/bin/env python3
name = "4h_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX: EMA(EMA(EMA(close,12),12),12) - 1 period percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100  # percent change
    trix = trix.fillna(0).values
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Choppiness regime filter: CHOP(14) > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
    high = prices['high'].values
    low = prices['low'].values
    atr14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    sum_tr = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    max_h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (max_h - min_l)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    chop_range = chop > 61.8  # ranging market
    chop_trending = chop < 38.2  # trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for TRIX and ATR
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trix[i]) or np.isnan(vol_ma20[i]) or np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX positive + volume spike + 1d uptrend + trending OR ranging regime
            if (trix[i] > 0 and volume_spike[i] and trend_up_1d_aligned[i] and
                (chop_trending[i] or chop_range[i])):
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative + volume spike + 1d downtrend + trending OR ranging regime
            elif (trix[i] < 0 and volume_spike[i] and not trend_up_1d_aligned[i] and
                  (chop_trending[i] or chop_range[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative OR 1d trend turns down
            if trix[i] < 0 or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive OR 1d trend turns up
            if trix[i] > 0 or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals