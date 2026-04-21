#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeSpike_v1
Hypothesis: 6h Ichimoku Kumo twist (Senkou Span A/B cross) with 1d EMA50 trend filter and volume confirmation (>1.6x 20-period MA).
Ichimoku cloud provides dynamic support/resistance and trend direction. Kumo twists indicate potential trend changes.
Uses ATR-based stop (2.0x) and minimum holding period of 4 bars to reduce churn.
Designed for 6h timeframe with 1d HTF trend to work in both bull and bear markets by requiring alignment with higher timeframe trend and strong volume confirmation.
Target: 75-150 total trades over 4 years (19-38/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (1.6x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 6h Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Kumo twist detection: Senkou Span A crosses Senkou Span B
    # Bullish twist: Senkou A crosses above Senkou B (previous A <= previous B and current A > current B)
    # Bearish twist: Senkou A crosses below Senkou B (previous A >= previous B and current A < current B)
    senkou_a_prev = np.roll(senkou_a, 1)
    senkou_b_prev = np.roll(senkou_b, 1)
    senkou_a_prev[0] = senkou_b_prev[0] = np.nan
    
    bullish_twist = (senkou_a > senkou_b) & (senkou_a_prev <= senkou_b_prev)
    bearish_twist = (senkou_a < senkou_b) & (senkou_a_prev >= senkou_b_prev)
    
    # Cloud top/bottom for filtering (price should be above/below cloud)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(bullish_twist[i]) or np.isnan(bearish_twist[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.6x average
        volume_confirm = volume_now > 1.6 * vol_avg
        
        if position == 0:
            # Long: bullish Kumo twist, price above cloud, above 1d EMA50, volume confirm
            long_condition = bullish_twist[i] and (price > cloud_top[i]) and (price > ema_50_1d_val) and volume_confirm
            # Short: bearish Kumo twist, price below cloud, below 1d EMA50, volume confirm
            short_condition = bearish_twist[i] and (price < cloud_bottom[i]) and (price < ema_50_1d_val) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below 1d EMA50)
                elif price < ema_50_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price re-enters cloud (loss of momentum)
                elif price < cloud_top[i] and price > cloud_bottom[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above 1d EMA50)
                elif price > ema_50_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price re-enters cloud (loss of momentum)
                elif price < cloud_top[i] and price > cloud_bottom[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0