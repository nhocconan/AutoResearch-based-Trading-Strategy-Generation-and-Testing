#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above upper BB AND 1d ADX > 25 AND volume > 1.5 * volume SMA(20)
# Short when price breaks below lower BB AND 1d ADX > 25 AND volume > 1.5 * volume SMA(20)
# Uses Bollinger Band width percentile to detect squeeze (low volatility breakout)
# Works in bull markets via longs in high ADX trends and bear markets via shorts in high ADX trends
# 12h timeframe reduces trade frequency to minimize fee drag while capturing medium-term breakouts
# Bollinger squeeze + ADX trend + volume confirmation provides high-probability entries

name = "12h_BollingerSqueeze_ADXTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Bollinger Bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Bollinger Bands (20, 2)
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Middle band = SMA(20)
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    # Upper and lower bands
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    # Bollinger Band Width
    bb_width = (upper_bb - lower_bb) / sma_20
    # BB Width percentile lookback 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    # Squeeze condition: BB Width < 20th percentile (low volatility)
    squeeze = bb_width_percentile < 20
    
    # Align Bollinger Bands and squeeze to prices timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb)
    squeeze_aligned = align_htf_to_ltf(prices, df_12h, squeeze.astype(float))
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Trend filter: ADX > 25 indicates strong trend
    strong_trend = adx > 25
    
    # Align ADX to prices timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend.astype(float))
    
    # Volume confirmation: volume > 1.5 * volume SMA(20) on 12h
    vol_12h = df_12h['volume'].values
    vol_sma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_12h > (1.5 * vol_sma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > upper BB AND squeeze AND strong trend AND volume spike
            if (close[i] > upper_bb_aligned[i] and 
                squeeze_aligned[i] > 0.5 and 
                strong_trend_aligned[i] > 0.5 and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < lower BB AND squeeze AND strong trend AND volume spike
            elif (close[i] < lower_bb_aligned[i] and 
                  squeeze_aligned[i] > 0.5 and 
                  strong_trend_aligned[i] > 0.5 and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < middle BB OR squeeze breaks (volatility expansion)
            if (close[i] < sma_20[-1] if len(sma_20) > 0 else 0) or squeeze_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > middle BB OR squeeze breaks (volatility expansion)
            if (close[i] > sma_20[-1] if len(sma_20) > 0 else 0) or squeeze_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals