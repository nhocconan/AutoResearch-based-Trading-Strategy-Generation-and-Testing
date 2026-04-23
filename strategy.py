#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
Long when: Tenkan-sen > Kijun-sen (TK cross bullish) AND price > Senkou Span A (cloud top) AND close > 1d EMA50 AND volume > 1.5x average.
Short when: Tenkan-sen < Kijun-sen (TK cross bearish) AND price < Senkou Span B (cloud bottom) AND close < 1d EMA50 AND volume > 1.5x average.
Exit when TK cross reverses OR price enters cloud OR volume drops below average.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-30 trades/year per symbol.
Ichimoku provides institutional support/resistance levels; 1d EMA50 filters for higher-timeframe trend; volume confirms conviction.
Works in bull markets via TK crosses above cloud and bear markets via short breakdowns below cloud with trend filter.
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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 plotted 26 periods ahead
    senkou_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 plotted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # The actual cloud boundaries for current period are Senkou Span A/B from 26 periods ago
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values are invalid due to lag
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top/bottom (Senkou A is always top in Ichimoku convention)
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        price = close[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        tk_bullish = tenkan > kijun
        tk_bearish = tenkan < kijun
        price_above_cloud = price > cloud_top_val
        price_below_cloud = price < cloud_bottom_val
        
        if position == 0:
            # Long: TK cross bullish AND price above cloud AND price > 1d EMA50 AND volume spike
            if (tk_bullish and price_above_cloud and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish AND price below cloud AND price < 1d EMA50 AND volume spike
            elif (tk_bearish and price_below_cloud and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: TK cross bearish OR price enters cloud OR volume drops below average
                if (not tk_bullish or price <= cloud_top_val or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: TK cross bullish OR price enters cloud OR volume drops below average
                if (not tk_bearish or price >= cloud_bottom_val or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0