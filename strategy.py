#!/usr/bin/env python3
# 6h_Ichimoku_Kumo_Twist_1dTrend_Volume
# Hypothesis: Ichimoku Kumo twist (Senkou Span A/B cross) with 1d trend filter and volume confirmation.
# Works in bull/bear: Kumo twist signals trend change, 1d trend filter ensures alignment with higher timeframe,
# volume confirms institutional participation. Avoids whipsaws in ranging markets.
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_Ichimoku_Kumo_Twist_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2, plotted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals to avoid look-ahead
    
    # Kumo twist: Senkou Span A crosses Senkou Span B
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    kumo_twist_bullish = (senkou_a > senkou_b) & (np.roll(senkou_a, 1) <= np.roll(senkou_b, 1))
    kumo_twist_bearish = (senkou_a < senkou_b) & (np.roll(senkou_a, 1) >= np.roll(senkou_b, 1))
    
    # Align Kumo twist signals to current timeframe (no look-ahead)
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kumo_twist_bullish.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kumo_twist_bearish.astype(float))
    
    # Get 1d trend: EMA 50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # Ensure Senkou B and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kumo_twist_bullish_aligned[i]) or np.isnan(kumo_twist_bearish_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish Kumo twist AND price above 1d EMA50 AND volume spike
            if (kumo_twist_bullish_aligned[i] > 0.5 and  # bullish twist signal
                close[i] > ema_50_1d_aligned[i] and
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish Kumo twist AND price below 1d EMA50 AND volume spike
            elif (kumo_twist_bearish_aligned[i] > 0.5 and  # bearish twist signal
                  close[i] < ema_50_1d_aligned[i] and
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish Kumo twist OR price below 1d EMA50
            if (kumo_twist_bearish_aligned[i] > 0.5 or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish Kumo twist OR price above 1d EMA50
            if (kumo_twist_bullish_aligned[i] > 0.5 or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals