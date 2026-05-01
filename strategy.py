#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume spike and chop regime filter
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trends when lines are aligned and separated.
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment).
# 1d volume spike confirms breakout strength. Chop regime filter avoids whipsaws in ranging markets.
# Designed for 12h timeframe to target 12-37 trades/year with discrete position sizing.
# Works in both bull (trend following) and bear (mean reversion via chop filter) markets.

name = "12h_WilliamsAlligator_1dVolumeSpike_ChopFilter_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for volume spike and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume spike: current volume > 2.0 * 20-period average volume
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (volume_ma_20_1d * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1d chop regime: Chopiness Index > 61.8 = ranging (mean revert), < 38.2 = trending
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chopiness Index(14) = 100 * log10(sum(ATR14) / (max(high)-min(low)) * sqrt(14))
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = (max_high_14 - min_low_14) * np.sqrt(14)
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid division by zero
    chop = 100 * np.log10(sum_atr_14 / chop_denominator)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA (smoothed moving average) of median price
    # Teeth: 8-period SMMA of median price
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(30, 13)  # need 30 for 1d indicators, 13 for Alligator jaw
    
    for i in range(start_idx, n):
        if (np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when chop < 50 (avoid strong ranging markets)
        regime_ok = chop_aligned[i] < 50
        
        # Volume confirmation
        vol_spike = volume_spike_1d_aligned[i]
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish alignment, volume spike, regime OK
            if bullish_alignment and vol_spike and regime_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment, volume spike, regime OK
            elif bearish_alignment and vol_spike and regime_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish alignment or loss of regime/volume conditions
            if bearish_alignment or not regime_ok or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish alignment or loss of regime/volume conditions
            if bullish_alignment or not regime_ok or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals