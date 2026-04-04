#!/usr/bin/env python3
"""
exp_6619_6h_ichimoku_cloud_1d_trend_v3
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter. Uses Ichimoku (Tenkan/Kijun/Senkou) on 6h for entry timing and 1d ADX > 25 for trend strength filter. Only takes longs when price above cloud AND 1d bullish trend, shorts when price below cloud AND 1d bearish trend. Ichimoku provides dynamic support/resistance while 1d ADX ensures we only trade in strong trends, avoiding choppy markets. Discrete sizing (0.25) minimizes fee churn. Includes ATR stoploss and max hold.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6619_6h_ichimoku_cloud_1d_trend_v3"
timeframe = "6h"
leverage = 1.0

# Parameters
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENKOU_B = 52
ADX_PERIOD = 14
ADX_THRESHOLD = 25
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 6  # ~6 * 6h = ~1.5 days

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=ADX_PERIOD, adjust=False).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    
    # Smoothed DM and TR
    dm_plus_smooth = dm_plus.ewm(span=ADX_PERIOD, adjust=False).mean().values
    dm_minus_smooth = dm_minus.ewm(span=ADX_PERIOD, adjust=False).mean().values
    tr_smooth = tr_1d.ewm(span=ADX_PERIOD, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, adjust=False).mean().values
    
    # Align ADX to LTF (6h) with shift(1) for completed bars only
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate LTF Ichimoku components
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_high = pd.Series(high).rolling(window=ICHIMOKU_TENKAN, min_periods=ICHIMOKU_TENKAN).max().values
    tenkan_low = pd.Series(low).rolling(window=ICHIMOKU_TENKAN, min_periods=ICHIMOKU_TENKAN).min().values
    tenkan = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_high = pd.Series(high).rolling(window=ICHIMOKU_KIJUN, min_periods=ICHIMOKU_KIJUN).max().values
    kijun_low = pd.Series(low).rolling(window=ICHIMOKU_KIJUN, min_periods=ICHIMOKU_KIJUN).min().values
    kijun = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    senkou_b_high = pd.Series(high).rolling(window=ICHIMOKU_SENKOU_B, min_periods=ICHIMOKU_SENKOU_B).max().values
    senkou_b_low = pd.Series(low).rolling(window=ICHIMOKU_SENKOU_B, min_periods=ICHIMOKU_SENKOU_B).min().values
    senkou_b = (senkou_b_high + senkou_b_low) / 2
    
    # Current cloud: Senkou Span A and B shifted back 26 periods
    senkou_a_shifted = np.roll(senkou_a, ICHIMOKU_KIJUN)
    senkou_b_shifted = np.roll(senkou_b, ICHIMOKU_KIJUN)
    # Fill first 26 values with NaN (not yet available)
    senkou_a_shifted[:ICHIMOKU_KIJUN] = np.nan
    senkou_b_shifted[:ICHIMOKU_KIJUN] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(ICHIMOKU_SENKOU_B, ICHIMOKU_KIJUN, ADX_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF or LTF data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(atr[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Determine trend from 1d ADX (trend strength, not direction)
        strong_trend = adx_aligned[i] > ADX_THRESHOLD
        
        # Ichimoku signals:
        # Price above cloud: bullish
        # Price below cloud: bearish
        # Tenkan/Kijun cross: momentum
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        # Enter new positions only if flat
        if position == 0:
            if strong_trend and price_above_cloud and tenkan_above_kijun:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif strong_trend and price_below_cloud and tenkan_below_kijun:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals