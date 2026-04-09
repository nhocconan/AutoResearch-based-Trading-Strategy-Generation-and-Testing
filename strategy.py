#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_trix_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for TRIX and volume
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Calculate daily TRIX (15-period)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1 period percent change
    close_d = df_d['close'].values
    ema1 = pd.Series(close_d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.full(len(close_d), np.nan)
    trix[14:] = (ema3[14:] - ema3[13:-1]) / ema3[13:-1] * 100  # percent change
    
    # Calculate daily volume moving average (20-period)
    vol_d = df_d['volume'].values
    vol_ma_20 = np.full(len(vol_d), np.nan)
    vol_sum = 0
    for i in range(len(vol_d)):
        vol_sum += vol_d[i]
        if i >= 20:
            vol_sum -= vol_d[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Load weekly data ONCE before loop for regime filter (ADX)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 30:
        return np.zeros(n)
    
    # Calculate weekly ADX (14-period) for regime filter
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # True Range
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_w[1:] - high_w[:-1]) > (low_w[:-1] - low_w[1:]), 
                       np.maximum(high_w[1:] - high_w[:-1], 0), 0)
    dm_minus = np.where((low_w[:-1] - low_w[1:]) > (high_w[1:] - high_w[:-1]), 
                        np.maximum(low_w[:-1] - low_w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = np.full(len(tr), np.nan)
    dm_plus_14 = np.full(len(dm_plus), np.nan)
    dm_minus_14 = np.full(len(dm_minus), np.nan)
    
    # Initial smoothed values (first 14 periods)
    if len(tr) >= 14:
        tr_14[13] = np.nansum(tr[1:15])  # skip first NaN
        dm_plus_14[13] = np.nansum(dm_plus[1:15])
        dm_minus_14[13] = np.nansum(dm_minus[1:15])
        
        # Wilder's smoothing
        for i in range(14, len(tr)):
            tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full(len(tr), np.nan)
    di_minus = np.full(len(tr), np.nan)
    dx = np.full(len(tr), np.nan)
    
    for i in range(14, len(tr)):
        if tr_14[i] != 0:
            di_plus[i] = (dm_plus_14[i] / tr_14[i]) * 100
            di_minus[i] = (dm_minus_14[i] / tr_14[i]) * 100
            dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    # ADX = smoothed DX
    adx = np.full(len(tr), np.nan)
    if len(dx) >= 27:  # need 14 + 13 for smoothing
        adx[26] = np.nansum(dx[14:28])  # first ADX value
        for i in range(27, len(dx)):
            adx[i] = adx[i-1] - (adx[i-1] / 14) + dx[i]
    
    # Align all indicators to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_d, trix)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_d, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TRIX turns negative OR low volatility regime
            if trix_aligned[i] < 0 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX turns positive OR low volatility regime
            if trix_aligned[i] > 0 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: TRIX positive AND high volume AND trending regime (ADX > 25)
            if (trix_aligned[i] > 0 and 
                volume[i] > vol_ma_20_aligned[i] * 1.5 and
                adx_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Enter short: TRIX negative AND high volume AND trending regime (ADX > 25)
            elif (trix_aligned[i] < 0 and 
                  volume[i] > vol_ma_20_aligned[i] * 1.5 and
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals