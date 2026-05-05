#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 1d ADX trend filter + volume confirmation
# Ichimoku identifies trend via Tenkan/Kijun cross and price relative to cloud
# Long when: Tenkan > Kijun AND price > cloud (Senkou Span A/B) AND 1d ADX > 25 AND volume > 1.5x 20-period MA
# Short when: Tenkan < Kijun AND price < cloud AND 1d ADX > 25 AND volume > 1.5x 20-period MA
# Exit when: Tenkan/Kijun cross reverses OR price re-enters cloud OR ADX drops below 20
# Uses Ichimoku for trend structure and support/resistance, ADX for trend strength, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_Ichimoku_1dADX_VolumeConfirm"
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
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    if len(high) >= period_tenkan:
        tenkan_sen = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                      pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    else:
        tenkan_sen = pd.Series(index=prices.index, dtype=float).values
        tenkan_sen[:] = np.nan
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    if len(high) >= period_kijun:
        kijun_sen = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                     pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    else:
        kijun_sen = pd.Series(index=prices.index, dtype=float).values
        kijun_sen[:] = np.nan
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    if len(tenkan_sen) >= period_kijun and not np.all(np.isnan(tenkan_sen)) and not np.all(np.isnan(kijun_sen)):
        senkou_a = ((tenkan_sen + kijun_sen) / 2)
    else:
        senkou_a = pd.Series(index=prices.index, dtype=float).values
        senkou_a[:] = np.nan
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    if len(high) >= period_senkou_b:
        senkou_b = (pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                    pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    else:
        senkou_b = pd.Series(index=prices.index, dtype=float).values
        senkou_b[:] = np.nan
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data for ADX
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # prepend NaN for first element
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM
        tr_period = 14
        atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False).mean().values
    else:
        adx = np.full(len(high_1d), np.nan)
    
    # ADX trend strength
    adx_strong = np.zeros(len(adx), dtype=bool)
    adx_weak = np.zeros(len(adx), dtype=bool)
    for i in range(len(adx)):
        if not np.isnan(adx[i]):
            adx_strong[i] = adx[i] > 25
            adx_weak[i] = adx[i] < 20
    
    # Align 1d ADX to 6h timeframe
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong.astype(float))
    adx_weak_aligned = align_htf_to_ltf(prices, df_1d, adx_weak.astype(float))
    
    # Ichimoku signals
    tenkan_above_kijun = tenkan_sen > kijun_sen
    tenkan_below_kijun = tenkan_sen < kijun_sen
    
    # Price above cloud (both Senkou Span A and B)
    price_above_cloud = (close > senkou_a) & (close > senkou_b)
    # Price below cloud (both Senkou Span A and B)
    price_below_cloud = (close < senkou_a) & (close < senkou_b)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # warmup for Ichimoku calculations
        # Skip if any value is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(adx_strong_aligned[i]) or np.isnan(adx_weak_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish Tenkan/Kijun cross + price above cloud + strong ADX + volume filter
            if (tenkan_above_kijun[i] and 
                price_above_cloud[i] and 
                adx_strong_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish Tenkan/Kijun cross + price below cloud + strong ADX + volume filter
            elif (tenkan_below_kijun[i] and 
                  price_below_cloud[i] and 
                  adx_strong_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish Tenkan/Kijun cross OR price re-enters cloud OR weak ADX
            if (tenkan_below_kijun[i] or 
                not price_above_cloud[i] or 
                adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish Tenkan/Kijun cross OR price re-enters cloud OR weak ADX
            if (tenkan_above_kijun[i] or 
                not price_below_cloud[i] or 
                adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals