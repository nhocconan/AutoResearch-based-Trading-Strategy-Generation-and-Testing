#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel breakout with 1w ADX regime filter
# Donchian breakout captures momentum; ADX > 25 confirms trending market
# In trending regime: follow breakout direction (long on upper band break, short on lower band break)
# In ranging regime (ADX < 20): mean revert at Donchian extremes (long at lower band, short at upper band)
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: trend following in strong trends, mean reversion in ranging markets

name = "12h_1d_1w_donchian_adx_regime_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    lookback = 20
    upper_1d = np.full(len(high_1d), np.nan)
    lower_1d = np.full(len(low_1d), np.nan)
    
    for i in range(lookback, len(high_1d)):
        upper_1d[i] = np.max(high_1d[i-lookback+1:i+1])
        lower_1d[i] = np.min(low_1d[i-lookback+1:i+1])
    
    # Calculate 1d ATR(14) for volatility normalization
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Normalize Donchian width by ATR to make it comparable across volatility regimes
    donchian_width_1d = upper_1d - lower_1d
    norm_width_1d = np.where(atr_1d > 0, donchian_width_1d / atr_1d, 0)
    
    # Load 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14)
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
    
    adx_1w = calculate_dmi(high_1w, low_1w, close_1w, 14)
    
    # Align 1d indicators to 12h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    norm_width_1d_aligned = align_htf_to_ltf(prices, df_1d, norm_width_1d)
    
    # Align 1w ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Threshold for Donchian breakout confirmation
    breakout_threshold = 0.5  # 0.5 ATR above/below bands
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or
            np.isnan(norm_width_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on 1w ADX
        trending_regime = adx_1w_aligned[i] > 25
        ranging_regime = adx_1w_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price falls below lower Donchian band
                if close[i] < lower_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price returns to mid-band
                mid_band = (upper_1d_aligned[i] + lower_1d_aligned[i]) / 2
                if close[i] > mid_band:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price rises above upper Donchian band
                if close[i] > upper_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price returns to mid-band
                mid_band = (upper_1d_aligned[i] + lower_1d_aligned[i]) / 2
                if close[i] < mid_band:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Follow Donchian breakout in trending market
                if close[i] > upper_1d_aligned[i] + breakout_threshold * atr_1d[-1] if len(atr_1d) > 0 and not np.isnan(atr_1d[-1]) else upper_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lower_1d_aligned[i] - breakout_threshold * atr_1d[-1] if len(atr_1d) > 0 and not np.isnan(atr_1d[-1]) else lower_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at Donchian extremes in ranging market
                if close[i] < lower_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > upper_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals