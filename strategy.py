#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1w EMA50 trend filter and volume confirmation.
Long when Tenkan-sen crosses above Kijun-sen AND price is above Cloud (Senkou Span A/B) 
AND 1w EMA50 is rising AND volume > 1.5x 20-period average.
Short when Tenkan-sen crosses below Kijun-sen AND price is below Cloud 
AND 1w EMA50 is falling AND volume > 1.5x 20-period average.
Exit when price crosses Tenkan-sen/Kijun-sen in opposite direction or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Targets 12-25 trades/year per symbol (48-100 total over 4 years) by requiring multiple confluence factors.
Designed to work in both bull and bear markets by trading with the 1w trend and using Ichimoku's
inherent trend/filter properties.
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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2.0
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind (not used for signals)
    
    # Align Ichimoku components (no additional delay needed as they are concurrent)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # EMA slope (rising/falling) - compare current vs 3 periods ago
    ema_slope = np.zeros_like(ema_1w_50_aligned)
    ema_slope[3:] = ema_1w_50_aligned[3:] - ema_1w_50_aligned[:-3]
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 52, 26, 9, 50, 20, 14, 3)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_1w_50_aligned[i]) or np.isnan(ema_slope[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_slope_val = ema_slope[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Tenkan/Kijun crossover detection
        tenkan_prev = tenkan_aligned[i-1] if i > 0 else tenkan_val
        kijun_prev = kijun_aligned[i-1] if i > 0 else kijun_val
        tenkan_cross_above = tenkan_val > kijun_val and tenkan_prev <= kijun_prev
        tenkan_cross_below = tenkan_val < kijun_val and tenkan_prev >= kijun_prev
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud AND 1w EMA50 rising AND volume spike
            if (tenkan_cross_above and 
                price > cloud_top and 
                ema_slope_val > 0 and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Tenkan crosses below Kijun AND price below cloud AND 1w EMA50 falling AND volume spike
            elif (tenkan_cross_below and 
                  price < cloud_bottom and 
                  ema_slope_val < 0 and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Tenkan/Kijun cross in opposite direction
            if position == 1 and tenkan_cross_below:
                exit_signal = True
            elif position == -1 and tenkan_cross_above:
                exit_signal = True
            
            # ATR-based stoploss: 2.5 * ATR from entry
            if position == 1 and price < entry_price - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_1wEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0