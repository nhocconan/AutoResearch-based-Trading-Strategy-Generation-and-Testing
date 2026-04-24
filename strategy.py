#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with weekly trend filter and volume confirmation.
- Long when Tenkan-sen crosses above Kijun-sen AND price > Senkou Span A (cloud top) AND weekly close > weekly EMA50
- Short when Tenkan-sen crosses below Kijun-sen AND price < Senkou Span B (cloud bottom) AND weekly close < weekly EMA50
- Volume must be > 1.3x ATR(14) * close (volatility-adjusted volume filter)
- Exit on opposite Ichimoku cross or weekly trend reversal
- Uses 6h primary timeframe with 1w HTF to target 50-150 trades over 4 years (12-37/year)
- Ichimoku provides dynamic support/resistance via cloud and momentum via TK cross
- Weekly EMA50 ensures alignment with major trend to avoid counter-trend whipsaws
- ATR-scaled volume filter adapts to changing volatility, reducing false breakouts
- Designed for BTC/ETH with edge in trending markets via cloud breaks and momentum confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods) using previous data (no look-ahead)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = high_series.rolling(window=9, min_periods=9).max().shift(1)
    period9_low = low_series.rolling(window=9, min_periods=9).min().shift(1)
    tenkan_sen = ((period9_high + period9_low) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = high_series.rolling(window=26, min_periods=26).max().shift(1)
    period26_low = low_series.rolling(window=26, min_periods=26).min().shift(1)
    kijun_sen = ((period26_high + period26_low) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = high_series.rolling(window=52, min_periods=52).max().shift(1)
    period52_low = low_series.rolling(window=52, min_periods=52).min().shift(1)
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for dynamic volume threshold
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 1.3 * ATR * close (volatility-adjusted)
    vol_threshold = 1.3 * atr * close
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (need 52+26 for Senkou B)
    start_idx = max(52 + 26, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku TK Cross signals
        tk_cross_up = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_down = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        if position == 0:
            # Long: TK cross up, price above cloud (Senkou A), weekly bullish, volume confirmation
            if (tk_cross_up and close[i] > senkou_a[i] and 
                close[i] > ema_50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross down, price below cloud (Senkou B), weekly bearish, volume confirmation
            elif (tk_cross_down and close[i] < senkou_b[i] and 
                  close[i] < ema_50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross down OR price falls below cloud OR weekly trend turns bearish
            if (tk_cross_down or close[i] < senkou_a[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross up OR price rises above cloud OR weekly trend turns bullish
            if (tk_cross_up or close[i] > senkou_b[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuTK_1wEMA50_ATRVolConfirm_v1"
timeframe = "6h"
leverage = 1.0