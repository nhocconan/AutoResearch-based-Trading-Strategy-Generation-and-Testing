#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter
# Bull power (high - EMA13) and Bear power (EMA13 - low) measure buying/selling pressure.
# In strong trends (ADX > 25): trade in direction of Elder Ray extreme (bull power > 0 for long, bear power < 0 for short).
# In weak trends/ranges (ADX <= 25): fade Elder Ray extremes (bull power < 0 for long, bear power > 0 for short).
# Volume confirmation reduces false signals. Discrete sizing 0.25 limits trades to ~12-37/year.
# Works in bull/bear markets: adapts to trend strength via ADX regime filter.

name = "6h_12h_elder_ray_adx_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA13 for Elder Ray
    close_s_12h = pd.Series(close_12h)
    ema13_12h = close_s_12h.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 12h Bull Power and Bear Power
    bull_power_12h = high_12h - ema13_12h
    bear_power_12h = ema13_12h - low_12h
    
    # Calculate 12h ADX(14)
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di_12h = np.where(atr_12h > 0, 100 * plus_dm_smooth / atr_12h, 0.0)
    minus_di_12h = np.where(atr_12h > 0, 100 * minus_dm_smooth / atr_12h, 0.0)
    
    # Calculate DX and ADX
    dx_12h = np.where((plus_di_12h + minus_di_12h) > 0, 
                      100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h), 
                      0.0)
    adx_12h = wilders_smoothing(dx_12h, 14)
    
    # Align 12h indicators to 6h timeframe
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Pre-compute volume confirmation array (6h volume > 1.5 * 20-period average)
    avg_volume_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * avg_volume_6h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_12h_aligned[i]) or np.isnan(bear_power_12h_aligned[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on ADX
        strong_trend = adx_12h_aligned[i] > 25.0
        weak_trend = adx_12h_aligned[i] <= 25.0
        
        if position == 1:  # Long position
            if strong_trend:
                # Exit long if bull power turns negative or trend weakens
                if bull_power_12h_aligned[i] < 0 or weak_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # weak_trend
                # Exit long if bull power becomes positive (fade the move) or trend strengthens
                if bull_power_12h_aligned[i] > 0 or strong_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if strong_trend:
                # Exit short if bear power turns negative or trend weakens
                if bear_power_12h_aligned[i] < 0 or weak_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # weak_trend
                # Exit short if bear power becomes positive (fade the move) or trend strengthens
                if bear_power_12h_aligned[i] > 0 or strong_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if strong_trend:
                # Enter long on positive bull power with volume confirmation
                if bull_power_12h_aligned[i] > 0 and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on positive bear power with volume confirmation
                elif bear_power_12h_aligned[i] > 0 and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            else:  # weak_trend
                # Enter long on negative bull power (fade) with volume confirmation
                if bull_power_12h_aligned[i] < 0 and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on negative bear power (fade) with volume confirmation
                elif bear_power_12h_aligned[i] < 0 and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals