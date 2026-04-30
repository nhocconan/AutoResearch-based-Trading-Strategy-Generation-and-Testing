#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d ADX trend filter and volume confirmation
# Ichimoku provides dynamic support/resistance via Kumo (cloud) and momentum via TK cross.
# In bull markets: price above cloud + TK cross up = long. In bear markets: price below cloud + TK cross down = short.
# 1d ADX > 25 filters for trending markets only, avoiding whipsaws in ranges.
# Volume spike confirms breakout validity. Discrete sizing 0.25 minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Ichimoku_Cloud_ADX_VolumeSpike_v1"
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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
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
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    # Not used for signals to avoid look-ahead
    
    # Kumo (cloud) boundaries: max/min of Senkou Span A & B
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # TK Cross: Tenkan-sen crossing Kijun-sen
    tk_cross = tenkan_sen - kijun_sen
    tk_cross_above = tk_cross > 0
    tk_cross_below = tk_cross < 0
    
    # Calculate 1d ADX for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(df_1d['high']).diff()
    dm_minus = -pd.Series(df_1d['low']).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr)
    di_minus = 100 * (dm_minus_smooth / atr)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align Ichimoku and ADX to 6h timeframe (wait for completed bars)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    upper_cloud_aligned = align_htf_to_ltf(prices, df_1d, upper_cloud)
    lower_cloud_aligned = align_htf_to_ltf(prices, df_1d, lower_cloud)
    tk_cross_above_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_above.astype(float))
    tk_cross_below_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_below.astype(float))
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(52, 20)  # warmup for Ichimoku (52) and volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(upper_cloud_aligned[i]) or np.isnan(lower_cloud_aligned[i]) or
            np.isnan(tk_cross_above_aligned[i]) or np.isnan(tk_cross_below_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_tenkan = tenkan_aligned[i]
        curr_kijun = kijun_aligned[i]
        curr_upper_cloud = upper_cloud_aligned[i]
        curr_lower_cloud = lower_cloud_aligned[i]
        curr_tk_cross_above = tk_cross_above_aligned[i] > 0.5
        curr_tk_cross_below = tk_cross_below_aligned[i] > 0.5
        curr_adx = adx_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Trend filter: require ADX > 25 for trending market
        if curr_adx <= 25:
            # In ranging markets, stay flat to avoid whipsaws
            signals[i] = 0.0
            position = 0
            continue
            
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price above cloud AND TK cross up
                if (curr_close > curr_upper_cloud and 
                    curr_tk_cross_above):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price below cloud AND TK cross down
                elif (curr_close < curr_lower_cloud and 
                      curr_tk_cross_below):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below cloud OR TK cross down
            if (curr_close < curr_lower_cloud or 
                curr_tk_cross_below):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above cloud OR TK cross up
            if (curr_close > curr_upper_cloud or 
                curr_tk_cross_above):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals