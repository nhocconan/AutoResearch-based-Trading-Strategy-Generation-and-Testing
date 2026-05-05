#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + Chop Regime Filter
# Long when Alligator jaws < teeth < lips (bullish alignment) AND volume spike AND chop < 61.8 (trending)
# Short when Alligator jaws > teeth > lips (bearish alignment) AND volume spike AND chop < 61.8 (trending)
# Williams Alligator uses smoothed medians (SMMA) with specific periods: jaws=13, teeth=8, lips=5
# Volume spike requires 2.0x 20-bar MA for confirmation
# Chop regime filter avoids whipsaws in ranging markets (CHOP > 61.8 = range, CHOP < 38.2 = trend)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in bull (trend following with Alligator alignment) and bear (avoids false signals via chop filter)
# Timeframe: 12h (as required)

name = "12h_WilliamsAlligator_1dVolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Alligator and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d (SMMA = smoothed moving average)
    # Jaws: SMMA(13, 8) - median price smoothed with period 13, shifted 8 bars
    # Teeth: SMMA(8, 5) - median price smoothed with period 8, shifted 5 bars
    # Lips: SMMA(5, 3) - median price smoothed with period 5, shifted 3 bars
    median_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    # Smoothed Moving Average (SMMA) = EMA with alpha = 1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 1.0 / period
        result = np.full_like(arr, np.nan)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result
    
    jaws_1d = smma(median_price_1d, 13)
    teeth_1d = smma(median_price_1d, 8)
    lips_1d = smma(median_price_1d, 5)
    
    # Shift jaws by 8, teeth by 5, lips by 3 (Alligator specific shifts)
    jaws_1d = np.roll(jaws_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    
    # Align Alligator lines to 12h timeframe
    jaws_1d_aligned = align_htf_to_ltf(prices, df_1d, jaws_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Volume confirmation on 12h (2.0x 20-bar MA)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    # Chop regime filter on 1d (Ehler's Chopiness Index)
    # CHOP = 100 * log10(sum(ATR(1), n) / (n * max(high-low, n))) / log10(n)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        if len(high_arr) < period + 1:
            return np.full_like(high_arr, np.nan)
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # align with original arrays
        
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        max_high_low = pd.Series(high_arr - low_arr).rolling(window=period, min_periods=period).max().values
        
        sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
        sum_max_hl = pd.Series(max_high_low).rolling(window=period, min_periods=period).sum().values
        
        chop = np.where(
            (sum_max_hl > 0) & ~np.isnan(sum_atr) & ~np.isnan(sum_max_hl),
            100 * np.log10(sum_atr / sum_max_hl) / np.log10(period),
            50.0  # default to neutral when undefined
        )
        return chop
    
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaws_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator alignment: jaws < teeth < lips
            bullish = jaws_1d_aligned[i] < teeth_1d_aligned[i] < lips_1d_aligned[i]
            # Bearish Alligator alignment: jaws > teeth > lips
            bearish = jaws_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i]
            # Trending regime: chop < 61.8
            trending = chop_1d_aligned[i] < 61.8
            
            if bullish and volume_spike[i] and trending:
                signals[i] = 0.25
                position = 1
            elif bearish and volume_spike[i] and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator loses bullish alignment OR chop > 61.8 (ranging)
            bullish = jaws_1d_aligned[i] < teeth_1d_aligned[i] < lips_1d_aligned[i]
            trending = chop_1d_aligned[i] < 61.8
            if not bullish or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator loses bearish alignment OR chop > 61.8 (ranging)
            bearish = jaws_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i]
            trending = chop_1d_aligned[i] < 61.8
            if not bearish or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals