#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 1d ADX trend filter + volume confirmation
# Ichimoku provides dynamic support/resistance via Kumo (cloud) and momentum via TK cross
# Long when price > Kumo and Tenkan > Kijun (bullish TK cross), Short when price < Kumo and Tenkan < Kijun (bearish TK cross)
# 1d ADX > 25 ensures strong trend direction (avoids ranging markets)
# Volume spike (1.8x 30-period average) confirms momentum strength
# Discrete sizing 0.25 minimizes fee churn. Target: 80-120 total trades over 4 years (20-30/year).

name = "6h_Ichimoku_1dADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Ichimoku components (6h timeframe)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # For plotting, Senkou spans are shifted forward 26 periods, but for current price comparison:
    # We use current Senkou A/B to determine if price is above/below cloud
    # Kumo cloud: price > max(Senkou A, Senkou B) = above cloud, price < min(Senkou A, Senkou B) = below cloud
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # Volume confirmation: volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(period_kijun, period_senkou_b, 14)  # warmup for Kijun, Senkou B, and 1d ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(kumo_top[i]) or 
            np.isnan(kumo_bottom[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_kumo_top = kumo_top[i]
        curr_kumo_bottom = kumo_bottom[i]
        curr_adx = adx_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require strong trend (ADX > 25) and volume spike
            if curr_adx > 25 and curr_volume_spike:
                # Bullish entry: price above cloud AND Tenkan > Kijun (bullish TK cross)
                if curr_close > curr_kumo_top and curr_tenkan > curr_kijun:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price below cloud AND Tenkan < Kijun (bearish TK cross)
                elif curr_close < curr_kumo_bottom and curr_tenkan < curr_kijun:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price falls below cloud OR Tenkan < Kijun (TK cross bearish)
            if curr_close < curr_kumo_top or curr_tenkan < curr_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above cloud OR Tenkan > Kijun (TK cross bullish)
            if curr_close > curr_kumo_bottom or curr_tenkan > curr_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals