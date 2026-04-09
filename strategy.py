#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Donchian breakout + volume confirmation + 1d chop regime filter
# Donchian(20) on 1w provides major trend structure and breakout signals
# Long when price breaks above 1w Donchian upper with volume confirmation and chop regime < 61.8 (trending)
# Short when price breaks below 1w Donchian lower with volume confirmation and chop regime < 61.8
# Exit when price returns to opposite Donchian band or chop regime > 61.8 (range)
# Uses discrete position sizing 0.25 to target ~20-40 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows major trends, chop filter avoids whipsaws in ranging markets

name = "12h_1w_donchian_breakout_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_upper_20 = rolling_max(high_1w, 20)
    donchian_lower_20 = rolling_min(low_1w, 20)
    
    # Calculate 1w average volume (20-period)
    vol_s_1w = pd.Series(volume_1w)
    avg_vol_1w = vol_s_1w.rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Chopiness Index (14-period)
    def calculate_atr(high, low, close, period):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr = np.zeros_like(high)
        if len(high) >= period:
            atr[period-1] = np.mean(tr[1:period+1])  # Simple average for first value
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    def calculate_adx(high, low, close, period):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing
        def wilder_smooth(arr, period):
            smoothed = np.zeros_like(arr)
            if len(arr) >= period:
                smoothed[period-1] = np.mean(arr[1:period+1])
                for i in range(period, len(arr)):
                    smoothed[i] = (smoothed[i-1] * (period-1) + arr[i]) / period
            return smoothed
        
        atr = wilder_smooth(tr, period)
        plus_di = 100 * wilder_smooth(plus_dm, period) / (atr + 1e-10)
        minus_di = 100 * wilder_smooth(minus_dm, period) / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = wilder_smooth(dx, period)
        return adx, plus_di, minus_di
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    adx_1d, _, _ = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Chopiness Index: CHOP = 100 * log10(sum(ATR)/ (n * log(n))) / log10(n)
    # Simplified approximation: CHOP = 100 * log10(ATR_sum / (n * log10(n))) / log10(n)
    # We'll use a common approximation: CHOP = 100 * log10(atr_sum / (period * log10(period))) / log10(period)
    chop_1d = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        atr_sum = np.sum(atr_1d[i-13:i+1])  # Sum of last 14 ATR values
        if atr_sum > 0:
            chop_1d[i] = 100 * np.log10(atr_sum / (14 * np.log10(14))) / np.log10(14)
        else:
            chop_1d[i] = 50  # Neutral value when ATR sum is zero
    
    # Align 1w indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_20)
    avg_vol_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_vol_1w)
    
    # Align 1d chop to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(avg_vol_1w_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume (20-period)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Chop regime filter: trending market (CHOP < 61.8)
        trending_regime = chop_1d_aligned[i] < 61.8
        
        if position == 1:  # Long position
            # Exit long if price falls below Donchian lower OR chop regime becomes ranging
            if close[i] < donchian_lower_aligned[i] or not trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above Donchian upper OR chop regime becomes ranging
            if close[i] > donchian_upper_aligned[i] or not trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on Donchian breakout with volume confirmation and trending regime
            if close[i] > donchian_upper_aligned[i] and volume_confirmed and trending_regime:
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_lower_aligned[i] and volume_confirmed and trending_regime:
                position = -1
                signals[i] = -0.25
    
    return signals