#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d Camarilla breakout with volume and session filter
# In trending markets (4h ADX > 25): breakout above/below 1d H3/L3 with volume confirmation
# In ranging markets (4h ADX < 20): mean reversion at 1d H3/L3 levels
# Uses 4h/1d for signal direction, 1h only for entry timing with session filter (08-20 UTC)
# Discrete position sizing 0.20 to limit trades to 15-37/year and reduce fee drag
# Works in bull/bear: breakout catches trends, ADX regime filter avoids whipsaws

name = "1h_4h_1d_camarilla_breakout_adx_volume_session_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ADX(14) for regime filter
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full(len(high), np.nan)
        
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = np.concatenate([[np.nan], high[1:] - high[:-1]])
        down_move = np.concatenate([[np.nan], low[:-1] - low[1:]])
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing
        def wilders_smoothing(values, period):
            if len(values) < period:
                return np.full(len(values), np.nan)
            alpha = 1.0 / period
            result = np.full(len(values), np.nan)
            result[period-1] = np.nanmean(values[:period])
            for i in range(period, len(values)):
                result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
            return result
        
        tr_smoothed = wilders_smoothing(tr, period)
        plus_dm_smoothed = wilders_smoothing(plus_dm, period)
        minus_dm_smoothed = wilders_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smoothed / tr_smoothed
        minus_di = 100 * minus_dm_smoothed / tr_smoothed
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = wilders_smoothing(dx, period)
        
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volume normalization
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
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_1d = volume_1d > 1.5 * avg_volume_1d
    
    # Calculate 1d Camarilla pivot levels (based on prior day to avoid look-ahead)
    range_1d = high_1d - low_1d
    h3_1d = close_1d + 1.1 * range_1d
    l3_1d = close_1d - 1.1 * range_1d
    h4_1d = close_1d + 1.5 * range_1d
    l4_1d = close_1d - 1.5 * range_1d
    
    # Align 1d indicators to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    volume_confirmed_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmed_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(h3_1d_aligned[i]) or
            np.isnan(l3_1d_aligned[i]) or np.isnan(h4_1d_aligned[i]) or
            np.isnan(l4_1d_aligned[i]) or np.isnan(volume_confirmed_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter based on 4h ADX
        trending_regime = adx_4h_aligned[i] > 25
        ranging_regime = adx_4h_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below L3 or we enter ranging regime
                if close[i] < l3_1d_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            elif ranging_regime:
                # Exit long if price rises above H3
                if close[i] > h3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above H3 or we enter ranging regime
                if close[i] > h3_1d_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            elif ranging_regime:
                # Exit short if price drops below L3
                if close[i] < l3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
        else:  # Flat
            if trending_regime:
                # Enter long on breakout above H3 with volume confirmation
                if close[i] > h3_1d_aligned[i] and volume_confirmed_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Enter short on breakout below L3 with volume confirmation
                elif close[i] < l3_1d_aligned[i] and volume_confirmed_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
            elif ranging_regime:
                # Mean reversion: buy near L3, sell near H3
                if close[i] <= l3_1d_aligned[i] and volume_confirmed_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                elif close[i] >= h3_1d_aligned[i] and volume_confirmed_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals