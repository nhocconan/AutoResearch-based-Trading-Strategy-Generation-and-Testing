#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d volume spike and trend regime filter
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long: Bull Power > 0 and Bear Power improving (less negative) with volume confirmation in uptrend (ADX > 25)
# Short: Bear Power < 0 and Bull Power deteriorating (less positive) with volume confirmation in downtrend (ADX > 25)
# Flat: otherwise or when ADX < 20 (weak trend)
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: trend following with strength confirmation avoids whipsaws

name = "6h_1d_elder_ray_volume_adx_v4"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 13-period EMA for Elder Ray
    def ema(values, span):
        if len(values) < span:
            return np.full(len(values), np.nan)
        alpha = 2.0 / (span + 1)
        result = np.full(len(values), np.nan)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    ema13_1d = ema(close_1d, 13)
    
    # Elder Ray components
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Calculate ADX for trend strength
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
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM
    atr_1d = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di_1d = 100 * plus_dm_smooth / atr_1d
    minus_di_1d = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Calculate 1d average volume (20-period)
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Pre-compute volume confirmation array
    volume_confirmed = volume > 1.5 * avg_volume_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Trend regime filter
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] < 20
        
        if position == 1:  # Long position
            if strong_trend:
                # Exit long if Bull Power turns negative or Bear Power deteriorates
                if bull_power_1d_aligned[i] <= 0 or bear_power_1d_aligned[i] > bear_power_1d_aligned[i-1]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif weak_trend:
                # Exit long in weak trend
                position = 0
                signals[i] = 0.0
            else:
                # Moderate trend - hold if still bullish
                if bull_power_1d_aligned[i] <= 0 or bear_power_1d_aligned[i] > bear_power_1d_aligned[i-1]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if strong_trend:
                # Exit short if Bear Power turns positive or Bull Power improves
                if bear_power_1d_aligned[i] >= 0 or bull_power_1d_aligned[i] < bull_power_1d_aligned[i-1]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif weak_trend:
                # Exit short in weak trend
                position = 0
                signals[i] = 0.0
            else:
                # Moderate trend - hold if still bearish
                if bear_power_1d_aligned[i] >= 0 or bull_power_1d_aligned[i] < bull_power_1d_aligned[i-1]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if strong_trend:
                # Enter long on bullish momentum with volume confirmation
                if bull_power_1d_aligned[i] > 0 and bear_power_1d_aligned[i] < bear_power_1d_aligned[i-1] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on bearish momentum with volume confirmation
                elif bear_power_1d_aligned[i] < 0 and bull_power_1d_aligned[i] < bull_power_1d_aligned[i-1] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals