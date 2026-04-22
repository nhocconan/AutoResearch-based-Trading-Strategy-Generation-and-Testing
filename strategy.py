#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams Alligator with 1-day ADX trend filter and volume spike.
Long when Jaw < Teeth < Lips (bullish alignment) with 1-day ADX > 25 and volume spike.
Short when Jaw > Teeth > Lips (bearish alignment) with 1-day ADX > 25 and volume spike.
Exit when Alligator lines cross (Jaw-Teeth or Teeth-Lips crossover) or ADX < 20.
Williams Alligator identifies trend phases; 1-day ADX filters strong trends; volume spike confirms momentum.
Designed for low trade frequency by requiring multiple confirmations. Works in both bull and bear markets
by following strong trends only when ADX confirms trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator (SMMA-based)
    jaw_period, jaw_shift = 13, 8
    teeth_period, teeth_shift = 8, 5
    lips_period, lips_shift = 5, 3
    
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        sma = np.nansum(arr[:period]) / period
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, jaw_period)
    jaw = np.roll(jaw, jaw_shift)
    jaw[:jaw_shift] = np.nan
    
    teeth = smma(close, teeth_period)
    teeth = np.roll(teeth, teeth_shift)
    teeth[:teeth_shift] = np.nan
    
    lips = smma(close, lips_period)
    lips = np.roll(lips, lips_shift)
    lips[:lips_shift] = np.nan
    
    # Load 1-day data for ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        up = high[1:] - high[:-1]
        down = low[:-1] - low[1:]
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.full_like(tr, np.nan)
        atr[period-1] = np.nanmean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_di = 100 * np.full_like(tr, np.nan)
        minus_di = 100 * np.full_like(tr, np.nan)
        for i in range(period, len(tr)):
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm[i] / atr[i]
                minus_di[i] = 100 * minus_dm[i] / atr[i]
        dx = np.full_like(tr, np.nan)
        for i in range(period, len(tr)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        adx = np.full_like(tr, np.nan)
        adx[2*period-1] = np.nanmean(dx[period-1:2*period-1])
        for i in range(2*period, len(tr)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        # Prepend NaN for alignment
        return np.concatenate([np.full(period+1, np.nan), adx])
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for Alligator and ADX
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Alligator alignment
        bullish_aligned = jaw[i] < teeth[i] < lips[i]
        bearish_aligned = jaw[i] > teeth[i] > lips[i]
        
        # ADX trend strength
        strong_trend = adx_14_aligned[i] > 25
        weak_trend = adx_14_aligned[i] < 20
        
        if position == 0:
            # Long: Bullish alignment + strong trend + volume spike
            if bullish_aligned and strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + strong trend + volume spike
            elif bearish_aligned and strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Bearish crossover OR weak trend
                if jaw[i] > teeth[i] or teeth[i] > lips[i] or weak_trend:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bullish crossover OR weak trend
                if jaw[i] < teeth[i] or teeth[i] < lips[i] or weak_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_ADX_Trend_Volume"
timeframe = "4h"
leverage = 1.0