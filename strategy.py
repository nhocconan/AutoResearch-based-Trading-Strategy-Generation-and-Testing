#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_VolumeSpike
Hypothesis: 6h Ichimoku Tenkan-Kijun (TK) cross with price above/below 1d cloud for trend alignment,
filtered by 1d EMA50 trend and 6h volume spike (>2x average). Enter long on bullish TK cross
when price > 1d cloud and 1d uptrend; short on bearish TK cross when price < 1d cloud and 1d downtrend.
Exit on opposite TK cross or ATR(14) trailing stop (2.5*ATR). Designed for low trade frequency
(~12-37 trades/year) to minimize fee drag. Works in bull/bear via 1d trend/cloud filter and volume spike.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku cloud and EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # === 6h Ichimoku Components (Tenkan, Kijun, Senkou Span A/B) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Align Ichimoku components to 6h timeframe (use previous completed bar)
    tenkan_aligned = align_htf_to_ltf(prices, prices.index.to_series(), tenkan)  # already 6h
    kijun_aligned = align_htf_to_ltf(prices, prices.index.to_series(), kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices.index.to_series(), senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices.index.to_series(), senkou_b)
    
    # === 1d Ichimoku Cloud (Senkou Span A/B) for trend filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Senkou Span A: (Tenkan + Kijun) / 2 calculated on 1d
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2.0
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2.0
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2.0
    
    # Senkou Span B: (52-period high + 52-period low) / 2
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2.0
    
    # Align 1d cloud to 6h timeframe (wait for completed 1d bar)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # === 1d EMA50 for additional HTF trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) 
            or np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        if position == 0:
            # Volume spike: current volume > 2x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_spike = volume[i] > 2.0 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_bullish = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_bearish = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            
            # Long conditions: bullish TK cross, price above cloud, 1d uptrend, volume spike
            long_condition = (tk_bullish and 
                            price > cloud_top and 
                            price > ema_50_1d_aligned[i] and 
                            vol_spike)
            
            # Short conditions: bearish TK cross, price below cloud, 1d downtrend, volume spike
            short_condition = (tk_bearish and 
                             price < cloud_bottom and 
                             price < ema_50_1d_aligned[i] and 
                             vol_spike)
            
            # Entry logic
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: bearish TK cross or price closes below cloud
            elif (tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]) or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: bullish TK cross or price closes above cloud
            elif (tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]) or price > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0