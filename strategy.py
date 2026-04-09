#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime filter
# Elder Ray: Bull Power = high - EMA(13), Bear Power = low - EMA(13)
# ADX > 25 indicates trending market (use Elder Ray for direction)
# ADX <= 25 indicates ranging market (fade extreme Elder Ray values)
# Weekly pivot high/low as dynamic support/resistance for breakout/mean reversion
# Designed for 6h timeframe: ~12-37 trades/year target
# Works in bull/bear markets: trend following in trending regimes, mean reversion in ranging regimes

name = "6h_1w_1d_elder_ray_adx_pivot_v1"
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
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA(13) for Elder Ray ===
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === 1d ADX(14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # DI and ADX
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14), 
                  0.0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # === 1d Elder Ray ===
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # === 1w pivot points (prior week to avoid look-ahead) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align all 1d and 1w indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or
            np.isnan(s2_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        price = close[i]
        pivot = pivot_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        r2 = r2_1w_aligned[i]
        s2 = s2_1w_aligned[i]
        
        # Regime: ADX > 25 = trending, ADX <= 25 = ranging
        trending = adx > 25
        ranging = adx <= 25
        
        if position == 1:  # Long position
            if trending:
                # Exit long if bull power turns negative or ADX weakens
                if bull_power <= 0 or adx < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging:
                # Exit long if price reaches R1 or bear power turns positive
                if price >= r1 or bear_power >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            if trending:
                # Exit short if bear power turns positive or ADX weakens
                if bear_power >= 0 or adx < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging:
                # Exit short if price reaches S1 or bull power turns negative
                if price <= s1 or bull_power <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending:
                # Enter long on strong bull power, short on strong bear power
                if bull_power > 0 and bear_power < 0:
                    position = 1
                    signals[i] = 0.25
                elif bear_power > 0 and bull_power < 0:
                    position = -1
                    signals[i] = -0.25
            elif ranging:
                # Mean reversion: buy near S1/S2, sell near R1/R2
                if price <= s2 and bear_power < 0:
                    position = 1
                    signals[i] = 0.25
                elif price >= r2 and bull_power > 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals