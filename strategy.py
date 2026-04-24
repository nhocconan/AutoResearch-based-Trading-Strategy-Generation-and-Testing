#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d EMA50 trend filter and volume confirmation.
- Long when Tenkan-sen crosses above Kijun-sen, price > Cloud (bullish), close > 1d EMA50, and volume confirmation
- Short when Tenkan-sen crosses below Kijun-sen, price < Cloud (bearish), close < 1d EMA50, and volume confirmation
- Uses 6h primary timeframe with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Ichimoku provides multiple confirmation lines (Tenkan, Kijun, Senkou Span A/B) reducing false signals
- 1d EMA50 ensures alignment with longer-term trend to avoid whipsaws in both bull and bear markets
- Volume confirmation filters low-momentum breakouts
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
    
    # Calculate Ichimoku components (using previous period data to avoid look-ahead)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = high_series.rolling(window=9, min_periods=9).max().shift(1)
    period9_low = low_series.rolling(window=9, min_periods=9).min().shift(1)
    tenkan_sen = ((period9_high + period9_low) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = high_series.rolling(window=26, min_periods=26).max().shift(1)
    period26_low = low_series.rolling(window=26, min_periods=26).min().shift(1)
    kijun_sen = ((period26_high + period26_low) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = high_series.rolling(window=52, min_periods=52).max().shift(1)
    period52_low = low_series.rolling(window=52, min_periods=52).min().shift(1)
    senkou_span_b = ((period52_high + period52_low) / 2).values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for dynamic volume threshold
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 1.5 * ATR * close (volatility-adjusted)
    vol_threshold = 1.5 * atr * close
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud color and position
        # Bullish cloud: Senkou Span A > Senkou Span B
        # Bearish cloud: Senkou Span A < Senkou Span B
        bullish_cloud = senkou_span_a[i] > senkou_span_b[i]
        bearish_cloud = senkou_span_a[i] < senkou_span_b[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > max(senkou_span_a[i], senkou_span_b[i])
        price_below_cloud = close[i] < min(senkou_span_a[i], senkou_span_b[i])
        
        # TK cross signals (using current vs previous)
        tk_cross_above = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_below = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        if position == 0:
            # Long: TK cross bullish, price above cloud, bullish trend, volume confirmation
            if tk_cross_above and price_above_cloud and bullish_cloud and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish, price below cloud, bearish trend, volume confirmation
            elif tk_cross_below and price_below_cloud and bearish_cloud and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross bearish OR price falls below cloud OR trend reverses
            if tk_cross_below or not price_above_cloud or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross bullish OR price rises above cloud OR trend reverses
            if tk_cross_above or not price_below_cloud or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuTK_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0