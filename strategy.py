#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 1d ADX trend filter + volume confirmation
# Ichimoku provides dynamic support/resistance via cloud and momentum via TK cross
# 1d ADX filters for trending regimes (ADX > 25) to avoid choppy markets
# Volume confirmation ensures breakout authenticity
# Works in bull/bear: ADX regime filter adapts to market conditions, Ichimoku cloud acts as dynamic S/R
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_1d_ichimoku_adx_volume_v1"
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
    
    # Load 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    # +DM, -DM, TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    tr_smoothed = wilders_smoothing(tr, 14)
    plus_dm_smoothed = wilders_smoothing(plus_dm, 14)
    minus_dm_smoothed = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used in signals to avoid look-ahead
    
    # Calculate 20-period average volume for volume confirmation
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
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        # Ichimoku signals
        # Bullish: price above cloud AND Tenkan > Kijun (TK cross up)
        # Bearish: price below cloud AND Tenkan < Kijun (TK cross down)
        # Cloud top/bottom: Senkou Span A/B (we use current values as cloud is plotted ahead)
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        tk_bullish = tenkan_sen[i] > kijun_sen[i]
        tk_bearish = tenkan_sen[i] < kijun_sen[i]
        
        # ADX filter: only trade in trending markets (ADX > 25)
        strong_trend = adx_aligned[i] > 25.0
        
        if position == 1:  # Long position
            # Exit: price falls below cloud OR TK cross turns bearish
            if close[i] < cloud_top or not tk_bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud OR TK cross turns bullish
            if close[i] > cloud_bottom or not tk_bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with Ichimoku, ADX filter, and volume confirmation
            if volume_confirmed and strong_trend:
                # Long entry: price above cloud AND TK cross bullish
                if price_above_cloud and tk_bullish:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price below cloud AND TK cross bearish
                elif price_below_cloud and tk_bearish:
                    position = -1
                    signals[i] = -0.25
    
    return signals