#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation
# Long when price breaks above R3 AND 1d close > 1d EMA34 (uptrend) AND volume > 1.8x 20 EMA
# Short when price breaks below S3 AND 1d close < 1d EMA34 (downtrend) AND volume > 1.8x 20 EMA
# Uses 4h for entry timing, 1d for trend direction to avoid counter-trend trades.
# Discrete sizing (0.25) to minimize fee churn. Target: 20-30 trades/year.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Added chop filter (CHOP > 61.8) to avoid whipsaws in ranging markets.

name = "4h_Camarilla_R3S3_1dTrend_VolumeConfirm_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h ATR for chop filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 14-period True Range sum for denominator
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Calculate 14-period high-low range for numerator
    hl_range = high - low
    hl_sum = pd.Series(hl_range).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum / hl_sum) / log10(14)
    # Avoid division by zero and log of zero
    chop = np.full(n, np.nan)
    mask = (hl_sum > 0) & (tr_sum > 0) & ~np.isnan(hl_sum) & ~np.isnan(tr_sum)
    chop[mask] = 100 * np.log10(tr_sum[mask] / hl_sum[mask]) / np.log10(14)
    
    # Chop regime: CHOP > 61.8 = ranging (avoid entries)
    chop_regime = chop > 61.8  # True when ranging, False when trending
    
    # Get 1d data for trend filter and Camarilla levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Get daily OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels for each day
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align daily Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 4h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(chop_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Avoid entries in ranging markets (chop > 61.8)
        if chop_regime[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND 1d uptrend AND volume spike
            if (close[i] > r3_aligned[i] and 
                uptrend_1d_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND 1d downtrend AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR 1d trend changes to downtrend
            if (close[i] < s3_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR 1d trend changes to uptrend
            if (close[i] > r3_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals