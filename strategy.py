#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + Volume Spike + Choppiness Regime
# TRIX (9) filters noise and captures momentum in trending markets
# Volume > 2.0x 20-period average confirms institutional interest
# Choppiness Index (14) < 38.2 identifies trending regimes (avoid range-bound whipsaws)
# Works in both bull/bear: TRIX adapts to momentum direction, volume confirms strength, chop filter avoids false signals
# Target: 20-30 trades/year to minimize fee drag while capturing strong momentum moves
# Focus on BTC/ETH as primary assets with proven TRIX effectiveness

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX and Choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate TRIX (9) on 1d closes
    close_1d = df_1d['close'].values
    # EMA1
    ema1 = pd.Series(close_1d).ewm(span=9, adjust=False, min_periods=9).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    # TRIX = 100 * (EMA3 - prev_EMA3) / prev_EMA3
    trix = np.full(len(close_1d), np.nan)
    trix[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix_smoothed = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_smoothed)
    
    # Calculate Choppiness Index (14) on 1d data
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    # Add first TR (high-low)
    tr = np.concatenate([[high[0] - low[0]], tr])
    
    # Sum of TR over 14 periods
    tr_sum = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        tr_sum[i] = np.sum(tr[i-14:i])
    
    # Highest high and lowest low over 14 periods
    max_high = np.full(len(high), np.nan)
    min_low = np.full(len(low), np.nan)
    for i in range(14, len(high)):
        max_high[i] = np.max(high[i-14:i])
        min_low[i] = np.min(low[i-14:i])
    
    # Chop = 100 * log10(sum(TR) / (max_high - min_low)) / log10(14)
    chop = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if tr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(20, vol_period)
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # TRIX momentum: positive = bullish, negative = bearish
        trix_bullish = trix_aligned[i] > 0
        trix_bearish = trix_aligned[i] < 0
        
        # Volume confirmation: spike > 2.0x average
        volume_confirmation = vol_ratio > 2.0
        
        # Choppiness regime: < 38.2 = trending (avoid chop > 61.8 = ranging)
        trending_regime = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long entry: TRIX bullish + volume spike + trending regime
            if trix_bullish and volume_confirmation and trending_regime:
                signals[i] = size
                position = 1
            # Short entry: TRIX bearish + volume spike + trending regime
            elif trix_bearish and volume_confirmation and trending_regime:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: TRIX turns bearish OR chop enters ranging regime
            if trix_aligned[i] <= 0 or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: TRIX turns bullish OR chop enters ranging regime
            if trix_aligned[i] >= 0 or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_TRIX_VolumeSpike_ChoppinessRegime"
timeframe = "4h"
leverage = 1.0