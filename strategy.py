#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 12h ADX trend confirmation and volume spike
# Uses Choppiness Index (14) to identify ranging markets (CHOP > 61.8) for mean reversion
# and trending markets (CHOP < 38.2) for trend following. ADX (12h) confirms trend strength.
# In ranging markets: buy near support (S1), sell near resistance (R1) using Camarilla levels.
# In trending markets: buy on pullbacks to EMA(21) in uptrend, sell on rallies to EMA(21) in downtrend.
# Volume spike (>1.5x average) confirms momentum. Designed for low trade frequency and high edge.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for ADX trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14) on 12h data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        for i in range(len(high)):
            if i < period:
                atr[i] = np.nan
            elif i == period:
                atr[i] = np.mean(tr[1:period+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        # Set first 'period' values to NaN
        adx[:period] = np.nan
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Load 1d data for Camarilla pivots (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14) on 4h data
    def calculate_choppiness(high, low, close, period=14):
        atr = np.zeros_like(high)
        for i in range(len(high)):
            if i == 0:
                atr[i] = high[i] - low[i]
            elif i < period:
                atr[i] = (atr[i-1] * i + (high[i] - low[i])) / (i + 1)
            else:
                atr[i] = (atr[i-1] * (period-1) + (high[i] - low[i])) / period
        
        sum_atr = np.zeros_like(high)
        for i in range(len(high)):
            if i < period:
                sum_atr[i] = np.sum(atr[1:i+1]) if i > 0 else 0
            else:
                sum_atr[i] = np.sum(atr[i-period+1:i+1])
        
        max_high = np.zeros_like(high)
        min_low = np.zeros_like(low)
        for i in range(len(high)):
            if i < period:
                max_high[i] = np.max(high[0:i+1])
                min_low[i] = np.min(low[0:i+1])
            else:
                max_high[i] = np.max(high[i-period+1:i+1])
                min_low[i] = np.min(low[i-period+1:i+1])
        
        chop = np.zeros_like(high)
        for i in range(len(high)):
            if i < period or max_high[i] == min_low[i]:
                chop[i] = np.nan
            else:
                chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(period)
        return chop
    
    chop = calculate_choppiness(high, low, close, 14)
    
    # Calculate Camarilla levels for previous day
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 4h EMA(21) for trend following pullbacks
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to avoid index issues
        # Skip if data not ready
        if (np.isnan(chop[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_21[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Ranging market (CHOP > 61.8): mean reversion at Camarilla levels
            if chop[i] > 61.8:
                # Long: Price near S1 support + volume spike
                if low[i] <= camarilla_s1_aligned[i] * 1.002 and volume[i] > 1.5 * vol_avg_20[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Price near R1 resistance + volume spike
                elif high[i] >= camarilla_r1_aligned[i] * 0.998 and volume[i] > 1.5 * vol_avg_20[i]:
                    signals[i] = -0.25
                    position = -1
            # Trending market (CHOP < 38.2): trend following pullbacks
            elif chop[i] < 38.2 and adx_12h_aligned[i] > 25:
                # Uptrend: price above EMA21, buy pullback to EMA
                if close[i] > ema_21[i]:
                    if low[i] <= ema_21[i] * 1.001 and volume[i] > 1.5 * vol_avg_20[i]:
                        signals[i] = 0.25
                        position = 1
                # Downtrend: price below EMA21, sell rally to EMA
                else:
                    if high[i] >= ema_21[i] * 0.999 and volume[i] > 1.5 * vol_avg_20[i]:
                        signals[i] = -0.25
                        position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below EMA21 OR reaches R1 resistance
                if close[i] < ema_21[i] * 0.999 or high[i] >= camarilla_r1_aligned[i] * 0.998:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price crosses above EMA21 OR reaches S1 support
                if close[i] > ema_21[i] * 1.001 or low[i] <= camarilla_s1_aligned[i] * 1.002:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Choppiness_ADX_Regime_Camarilla_EMA"
timeframe = "4h"
leverage = 1.0