#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Uses Ichimoku (Tenkan/Kijun/Senkou Span A/B) on 6h for entry signals
# 1d ADX > 25 ensures alignment with strong daily trend to avoid counter-trend whipsaws
# Volume spike (2.0x 96-period average) confirms institutional participation
# Cloud acts as dynamic support/resistance - price above cloud = bull bias, below = bear bias
# Tenkan-Kijun cross provides momentum confirmation within the trend
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via breakouts above cloud with TK cross up and bear markets via breakdowns below cloud with TK cross down.

name = "6h_Ichimoku_Cloud_1dADX25_VolumeSpike_v1"
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
    
    # Load 6h data for Ichimoku calculation (primary timeframe)
    df_6h = prices  # Use the passed prices DataFrame for 6h calculations
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    # Not used for signals as it requires future data
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr_1d = wilder_smooth(tr, period)
    plus_di_1d = 100 * wilder_smooth(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, period) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    # Handle division by zero
    dx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx_1d)
    adx_1d = wilder_smooth(dx_1d, period)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 2.0x 96-period average (96*6h = 576h = 24 days)
    vol_ma_96 = pd.Series(volume).rolling(window=96, min_periods=96).mean().values
    volume_spike = volume > (2.0 * vol_ma_96)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Ichimoku periods and volume MA
    start_idx = max(52, 96)  # Senkou B needs 52 periods, volume MA needs 96
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_96[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_senkou_a = senkou_a[i]
        curr_senkou_b = senkou_b[i]
        curr_adx = adx_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Determine cloud top and bottom
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        # Determine if price is above or below cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Tenkan-Kijun cross signals
        tk_cross_up = curr_tenkan > curr_kijun
        tk_cross_down = curr_tenkan < curr_kijun
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and strong trend (ADX > 25)
            if curr_volume_spike and curr_adx > 25:
                # Bullish entry: price above cloud AND Tenkan crosses above Kijun
                if price_above_cloud and tk_cross_up:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price below cloud AND Tenkan crosses below Kijun
                elif price_below_cloud and tk_cross_down:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price falls below cloud OR Tenkan crosses below Kijun
            if price_below_cloud or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above cloud OR Tenkan crosses above Kijun
            if price_above_cloud or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals