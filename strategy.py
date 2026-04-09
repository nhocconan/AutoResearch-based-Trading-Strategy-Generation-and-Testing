#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d regime filter
# Donchian(20) breakout provides clear trend direction signals
# Volume > 1.5x 20-period average confirms breakout strength
# 1d ADX < 20 triggers mean reversion at Donchian bands, ADX > 25 triggers trend following
# Uses discrete position sizing 0.20 to target ~15-37 trades/year on 1h timeframe
# Session filter (08-20 UTC) reduces noise outside active trading hours
# Works in bull/bear markets: trend following in strong trends, mean reversion in ranging markets

name = "1h_4h_1d_donchian_volume_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    open_time = prices['open_time']
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full(len(arr), np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full(len(arr), np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high_4h = rolling_max(high_4h, 20)
    donchian_low_4h = rolling_min(low_4h, 20)
    
    # Calculate 4h ATR(14) for Donchian band width
    tr1 = np.abs(high_4h[1:] - low_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_4h = wilders_smoothing(tr_4h, 14)
    
    # Calculate 4h Donchian width normalized by ATR
    donchian_width_4h = donchian_high_4h - donchian_low_4h
    donchian_width_norm_4h = np.where(atr_4h > 0, donchian_width_4h / atr_4h, 0)
    
    # Load 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
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
    
    adx_1d = calculate_dmi(high_1d, low_1d, close_1d, 14)
    
    # Align 4h indicators to 1h timeframe
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    donchian_width_norm_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_width_norm_4h)
    
    # Align 1d ADX to 1h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1h volume average (20-period) for volume confirmation
    volume_s = pd.Series(volume)
    volume_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(donchian_width_norm_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_ma_20[i]) or not in_session.iloc[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma_20[i]
        
        # Regime filter based on 1d ADX
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below Donchian low
                if close[i] <= donchian_low_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            elif ranging_regime:
                # Exit long if price returns to Donchian mean
                donchian_mid = (donchian_high_4h_aligned[i] + donchian_low_4h_aligned[i]) / 2
                if close[i] >= donchian_mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above Donchian high
                if close[i] >= donchian_high_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            elif ranging_regime:
                # Exit short if price returns to Donchian mean
                donchian_mid = (donchian_high_4h_aligned[i] + donchian_low_4h_aligned[i]) / 2
                if close[i] <= donchian_mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
        else:  # Flat
            if trending_regime and volume_confirmed:
                # Donchian breakout with volume confirmation in trending market
                if close[i] > donchian_high_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                elif close[i] < donchian_low_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
            elif ranging_regime:
                # Mean reversion at Donchian bands in ranging market
                if close[i] <= donchian_low_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                elif close[i] >= donchian_high_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals