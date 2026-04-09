#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation
# Uses 1d Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
# 1d ADX > 25 filters for trending markets to avoid false breakouts in ranging conditions
# Volume confirmation ensures breakout conviction
# Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: ADX adapts to regime, Camarilla provides structured levels

name = "6h_1d_camarilla_adx_volume_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla formula: Pivot = (H+L+C)/3
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Calculate Camarilla levels for 1d
    camarilla_pivot = typical_price.values
    camarilla_r4 = df_1d['close'].values + (range_hl * 1.1 / 2)
    camarilla_r3 = df_1d['close'].values + (range_hl * 1.1 / 4)
    camarilla_s3 = df_1d['close'].values - (range_hl * 1.1 / 4)
    camarilla_s4 = df_1d['close'].values - (range_hl * 1.1 / 2)
    
    # Align 1d Camarilla levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d ADX (14-period) for trend strength filter
    # ADX requires +DI, -DI, and TR calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth TR and DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 20-period average volume for volume confirmation (6h volume)
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(pivot_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(adx_6h[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_6h[i] > 25
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (mean reversion level) OR trend weakens
            if close[i] < s3_6h[i] or adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (mean reversion level) OR trend weakens
            if close[i] > r3_6h[i] or adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume and trend confirmation
            if volume_confirm and trending:
                # Long breakout: price closes above R4 AND above pivot (bullish breakout)
                if close[i] > r4_6h[i] and close[i] > pivot_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below S4 AND below pivot (bearish breakout)
                elif close[i] < s4_6h[i] and close[i] < pivot_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals