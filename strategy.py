#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels with volume confirmation and 1w ADX regime filter
# In trending markets (ADX > 25): breakout long when price > H3 with volume > 1.5x average volume
# In ranging markets (ADX < 20): mean revert at H3/L3 levels with volume confirmation
# Uses discrete position sizing 0.25 to limit trades to ~20-50/year and reduce fee drag
# Works in bull/bear markets: trend following in strong trends, mean reversion in ranging markets

name = "4h_1d_1w_camarilla_pivot_volume_adx_v1"
timeframe = "4h"
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
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # H4 = Close + Range * 1.1/2
    # H3 = Close + Range * 1.1/4
    # H2 = Close + Range * 1.1/6
    # H1 = Close + Range * 1.1/12
    # L1 = Close - Range * 1.1/12
    # L2 = Close - Range * 1.1/6
    # L3 = Close - Range * 1.1/4
    # L4 = Close - Range * 1.1/2
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    h3_1d = close_1d + range_1d * 1.1 / 4.0
    l3_1d = close_1d - range_1d * 1.1 / 4.0
    h4_1d = close_1d + range_1d * 1.1 / 2.0
    l4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Load 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[:-1])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
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
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Calculate average volume for confirmation (20-period SMA)
    def sma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        for i in range(period-1, len(values)):
            result[i] = np.mean(values[i-period+1:i+1])
        return result
    
    avg_volume_1d = sma(volume, 20)  # Using current timeframe volume as proxy for 1d volume confirmation
    
    # Align 1d indicators to 4h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Align 1w ADX to 4h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Align volume average to 4h timeframe
    avg_volume_aligned = align_htf_to_ltf(prices, prices, avg_volume_1d)  # Self-align for same timeframe
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(avg_volume_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_aligned[i]
        
        # Regime filter based on 1w ADX
        trending_regime = adx_1w_aligned[i] > 25
        ranging_regime = adx_1w_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price falls below H3
                if close[i] <= h3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price returns from H3 level
                if close[i] >= h3_1d_aligned[i] * 0.995:  # Slight buffer to avoid whipsaw
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price rises above L3
                if close[i] >= l3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price returns from L3 level
                if close[i] <= l3_1d_aligned[i] * 1.005:  # Slight buffer to avoid whipsaw
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime and volume_confirmed:
                # Breakout long when price > H3 with volume confirmation
                if close[i] > h3_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short when price < L3 with volume confirmation
                elif close[i] < l3_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime and volume_confirmed:
                # Mean revert at extreme levels: long at L3, short at H3
                if close[i] <= l3_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= h3_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals