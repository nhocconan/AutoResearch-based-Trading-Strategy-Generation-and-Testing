#!/usr/bin/env python3

"""
Hypothesis: 6-hour Ichimoku Cloud with 1-day/1-week trend filter.
Uses Tenkan/Kijun cross for entry, with Senkou Span cloud as support/resistance.
Trades in direction of 1-day and 1-week EMAs to align with higher timeframe trend.
Avoids trades when price is inside cloud (low probability). Designed for low trade frequency
(15-30 trades/year) to minimize fee flood. Works in bull/bear by following higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A/B, Chikou."""
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = (period52_high + period52_low) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku on 6h data
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Load daily data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily EMA for trend filter (50-period)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly EMA for trend filter (50-period)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Senkou B calculation
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # Volume confirmation
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0 and vol_filter:
            # Long: TK cross bullish AND price above cloud AND both timeframes bullish
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and  # TK cross bullish
                close[i] > cloud_top and 
                close[i] > ema_50_1d_aligned[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish AND price below cloud AND both timeframes bearish
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and  # TK cross bearish
                  close[i] < cloud_bottom and 
                  close[i] < ema_50_1d_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: TK cross in opposite direction OR price re-enters cloud
            exit_signal = False
            
            if position == 1:
                # Exit long: TK cross bearish OR price drops below cloud bottom
                if (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]) or close[i] < cloud_bottom:
                    exit_signal = True
            else:  # position == -1
                # Exit short: TK cross bullish OR price rises above cloud top
                if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]) or close[i] > cloud_top:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_1d1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0