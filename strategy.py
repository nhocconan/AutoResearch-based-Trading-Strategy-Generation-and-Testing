#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align pivot levels to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 14-period daily ATR for volatility filter
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike detection (20-period average on 4h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate 4h ADX for trend strength filter
    period = 14
    tr_4h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_4h[0] = high[0] - low[0]
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    atr_tr = np.full(n, np.nan)
    if n >= period:
        atr_tr[period-1] = np.mean(tr_4h[:period])
        for i in range(period, n):
            atr_tr[i] = (atr_tr[i-1] * (period-1) + tr_4h[i]) / period
    
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    if n >= 2*period:
        plus_sm = np.full(n, np.nan)
        minus_sm = np.full(n, np.nan)
        plus_sm[2*period-1] = np.sum(plus_dm[:2*period])
        minus_sm[2*period-1] = np.sum(minus_dm[:2*period])
        for i in range(2*period, n):
            plus_sm[i] = plus_sm[i-1] - (plus_sm[i-1] / period) + plus_dm[i]
            minus_sm[i] = minus_sm[i-1] - (minus_sm[i-1] / period) + minus_dm[i]
        plus_di = 100 * plus_sm / atr_tr
        minus_di = 100 * minus_sm / atr_tr
        dx = np.full(n, np.nan)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = np.full(n, np.nan)
        if n >= 3*period:
            adx[3*period-1] = np.mean(dx[2*period:3*period])
            for i in range(3*period, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size for lower drawdown
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_4h[i]) or 
            np.isnan(r1_4h[i]) or
            np.isnan(s1_4h[i]) or
            np.isnan(r2_4h[i]) or
            np.isnan(s2_4h[i]) or
            np.isnan(atr_4h[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.8% of price)
        if atr_4h[i] < 0.008 * close[i]:
            signals[i] = 0.0
            continue
        
        # Require strong trend (ADX > 25)
        if adx[i] < 25:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above R2 with volume confirmation and strong trend
            if (close[i] > r2_4h[i] and volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S2 with volume confirmation and strong trend
            elif (close[i] < s2_4h[i] and volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below S1
            if close[i] < s1_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above R1
            if close[i] > r1_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Pivot_R2S2_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0