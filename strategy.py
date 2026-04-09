#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using Williams Alligator (SMMA-based) with 1w ADX regime filter
# Alligator identifies trend via Jaw/Teeth/Lips SMMA alignment
# ADX > 25 = trending (follow Alligator direction), ADX < 20 = ranging (fade extreme Alligator misalignment)
# Discrete sizing 0.25 to limit trades to ~15-35/year and reduce fee drag
# Works in bull/bear: trend following in strong trends, mean reversion in ranging markets

name = "1d_1w_alligator_adx_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator using SMMA (Smoothed Moving Average)
    def smma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period - 1) + values[i]) / period
        return result
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Alligator trend indicators
    # Bullish: Lips > Teeth > Jaw
    # Bearish: Lips < Teeth < Jaw
    alligator_bull = (lips > teeth) & (teeth > jaw)
    alligator_bear = (lips < teeth) & (teeth < jaw)
    
    # Alligator misalignment strength for ranging regime
    # Normalized distance between Lips and Jaw
    jaw_lips_diff = lips - jaw
    close_s_1d = pd.Series(close_1d)
    atr_1d = close_s_1d.rolling(14, min_periods=14).std().values * np.sqrt(252)  # approximate ATR
    atr_1d = np.where(atr_1d == 0, 1e-10, atr_1d)  # avoid division by zero
    alligator_misalignment = np.where(~np.isnan(jaw_lips_diff), jaw_lips_diff / atr_1d, 0)
    
    # Load 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14) using Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = wilders_smoothing(tr, 14)
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1w
    minus_di = 100 * minus_dm_smooth / atr_1w
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1w = wilders_smoothing(dx, 14)
    
    # Align indicators to 1d timeframe
    alligator_bull_aligned = align_htf_to_ltf(prices, df_1d, alligator_bull.astype(float))
    alligator_bear_aligned = align_htf_to_ltf(prices, df_1d, alligator_bear.astype(float))
    alligator_misalignment_aligned = align_htf_to_ltf(prices, df_1d, alligator_misalignment)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Thresholds
    extreme_misalignment = 2.0  # 2 ATR deviations for fading in ranging market
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(alligator_bull_aligned[i]) or np.isnan(alligator_bear_aligned[i]) or
            np.isnan(alligator_misalignment_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on 1w ADX
        trending_regime = adx_1w_aligned[i] > 25
        ranging_regime = adx_1w_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if Alligator turns bearish
                if alligator_bear_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if misalignment returns from extreme
                if alligator_misalignment_aligned[i] > -extreme_misalignment * 0.5:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if Alligator turns bullish
                if alligator_bull_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if misalignment returns from extreme
                if alligator_misalignment_aligned[i] < extreme_misalignment * 0.5:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Follow Alligator signals in trending market
                if alligator_bull_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif alligator_bear_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Fade extreme Alligator misalignment in ranging market
                if alligator_misalignment_aligned[i] < -extreme_misalignment:
                    position = 1
                    signals[i] = 0.25
                elif alligator_misalignment_aligned[i] > extreme_misalignment:
                    position = -1
                    signals[i] = -0.25
    
    return signals