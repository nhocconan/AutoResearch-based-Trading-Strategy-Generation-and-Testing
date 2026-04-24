#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d ADX trend filter and volume confirmation.
- Long when Tenkan-sen crosses above Kijun-sen (bullish TK cross) AND price is above Cloud (bullish bias) AND 1d ADX > 25 (strong trend) AND volume > 1.5x ATR(14)*close
- Short when Tenkan-sen crosses below Kijun-sen (bearish TK cross) AND price is below Cloud (bearish bias) AND 1d ADX > 25 (strong trend) AND volume > 1.5x ATR(14)*close
- Exit when TK cross reverses OR price crosses opposite Cloud boundary (Tenkan-sen crosses opposite Kijun-sen OR price crosses Cloud)
- Uses 6h primary timeframe with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Ichimoku provides dynamic support/resistance via Cloud and momentum via TK cross
- 1d ADX filter ensures we only trade in strong trending markets, avoiding whipsaws in ranging/ bear markets
- ATR-scaled volume filter adapts to changing volatility, reducing false signals
- Designed for BTC/ETH with edge in both bull (breakout continuation) and bear (strong trend continuation) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components (using previous period to avoid look-ahead)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = high_series.rolling(window=9, min_periods=9).max().shift(1)
    period9_low = low_series.rolling(window=9, min_periods=9).min().shift(1)
    tenkan = ((period9_high + period9_low) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = high_series.rolling(window=26, min_periods=26).max().shift(1)
    period26_low = low_series.rolling(window=26, min_periods=26).min().shift(1)
    kijun = ((period26_high + period26_low) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = high_series.rolling(window=52, min_periods=52).max().shift(1)
    period52_low = low_series.rolling(window=52, min_periods=52).min().shift(1)
    senkou_b = ((period52_high + period52_low) / 2)
    # Shift both Senkou spans forward by 26 periods (cloud is plotted ahead)
    senkou_a = pd.Series(senkou_a).shift(26).values
    senkou_b = pd.Series(senkou_b).shift(26).values
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    dm_plus.iloc[0] = np.nan
    dm_minus.iloc[0] = np.nan
    
    # Smoothed values
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean()
    dm_plus_smooth = dm_plus.ewm(span=14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = dm_minus.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = dx.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate ATR(14) for dynamic volume threshold (6h)
    tr1_6h = pd.Series(high - low)
    tr2_6h = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3_6h = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2_6h.iloc[0] = np.nan
    tr3_6h.iloc[0] = np.nan
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h = tr_6h.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 1.5 * ATR * close (volatility-adjusted)
    vol_threshold = 1.5 * atr_6h * close
    volume_confirm = volume > vol_threshold
    
    # Cloud boundaries (top and bottom of cloud)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52+26, 30) + 1  # Senkou B needs 52+26 periods, ADX needs 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish TK cross: Tenkan crosses above Kijun
        bullish_tk_cross = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        # Bearish TK cross: Tenkan crosses below Kijun
        bearish_tk_cross = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        if position == 0:
            # Long: bullish TK cross, price above cloud, strong trend (ADX > 25), volume confirmation
            if bullish_tk_cross and close[i] > cloud_top[i] and adx_1d_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross, price below cloud, strong trend (ADX > 25), volume confirmation
            elif bearish_tk_cross and close[i] < cloud_bottom[i] and adx_1d_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish TK cross OR price crosses below cloud bottom
            if bearish_tk_cross or close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish TK cross OR price crosses above cloud top
            if bullish_tk_cross or close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuTK_1dADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0