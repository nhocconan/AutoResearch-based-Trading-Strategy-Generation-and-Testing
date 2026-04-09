#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using Williams Alligator (Jaw/Teeth/Lips) with 1w ADX regime filter
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3) - identifies trend strength and direction
# ADX > 25 indicates trending market; ADX < 20 indicates ranging market
# In trending regime (ADX > 25): follow Alligator alignment (long when Lips>Teeth>Jaw, short when Lips<Teeth<Jaw)
# In ranging regime (ADX < 20): mean revert at extreme Alligator deviations (long when Lips far below Jaw, short when Lips far above Jaw)
# Uses discrete position sizing 0.25 to limit trades to ~7-25/year and reduce fee drag
# Works in bull/bear markets: trend following in strong trends, mean reversion in ranging markets

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
    
    # Load 1d data ONCE before loop (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components on 1d
    # Jaw: SMA(13, 8) - slowest
    close_s_1d = pd.Series(close_1d)
    jaw_1d = close_s_1d.rolling(window=13, min_periods=13).mean().shift(8).values
    
    # Teeth: SMA(8, 5) - medium
    teeth_1d = close_s_1d.rolling(window=8, min_periods=8).mean().shift(5).values
    
    # Lips: SMA(5, 3) - fastest
    lips_1d = close_s_1d.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1d ATR(14) for volatility normalization
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Normalize Alligator deviations by ATR to make signals volatility-adjusted
    # Deviation = Lips - Jaw (positive = bullish alignment, negative = bearish alignment)
    lips_minus_jaw_1d = lips_1d - jaw_1d
    norm_deviation_1d = np.where(atr_1d > 0, lips_minus_jaw_1d / atr_1d, 0)
    
    # Load 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14)
    def calculate_dmi(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[:-1])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = wilders_smoothing(tr, period)
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM
        plus_dm_smooth = wilders_smoothing(plus_dm, period)
        minus_dm_smooth = wilders_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smoothing(dx, period)
        
        return adx
    
    adx_1w = calculate_dmi(high_1w, low_1w, close_1w, 14)
    
    # Align 1d indicators to 1d timeframe (no shift needed as we're already on 1d)
    norm_deviation_1d_aligned = align_htf_to_ltf(prices, df_1d, norm_deviation_1d)
    
    # Align 1w ADX to 1d timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Threshold for extreme deviations in ranging market
    extreme_threshold = 1.0  # 1.0 ATR deviation
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(norm_deviation_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on 1w ADX
        trending_regime = adx_1w_aligned[i] > 25
        ranging_regime = adx_1w_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if Alligator alignment breaks (Lips <= Jaw)
                if norm_deviation_1d_aligned[i] <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if deviation returns from extreme
                if norm_deviation_1d_aligned[i] > -extreme_threshold * 0.5:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if Alligator alignment breaks (Lips >= Jaw)
                if norm_deviation_1d_aligned[i] >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if deviation returns from extreme
                if norm_deviation_1d_aligned[i] < extreme_threshold * 0.5:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Follow Alligator alignment in trending market
                if norm_deviation_1d_aligned[i] > 0:
                    position = 1
                    signals[i] = 0.25
                elif norm_deviation_1d_aligned[i] < 0:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at extreme Alligator deviations in ranging market
                if norm_deviation_1d_aligned[i] < -extreme_threshold:
                    position = 1
                    signals[i] = 0.25
                elif norm_deviation_1d_aligned[i] > extreme_threshold:
                    position = -1
                    signals[i] = -0.25
    
    return signals