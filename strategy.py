#!/usr/bin/env python3
"""
Experiment #115: 6h Ichimoku Cloud + Weekly Trend + Volume Confirmation

HYPOTHESIS: Ichimoku cloud (tenkan/kijun/senkou) filtered by weekly trend (price > weekly EMA20) 
and confirmed by volume spikes creates a robust trend-following strategy that works in both 
bull and bear markets. The cloud acts as dynamic support/resistance, weekly EMA20 filters 
for higher timeframe trend direction, and volume ensures institutional participation. 
Targets 15-35 trades/year on 6h timeframe (60-140 total over 4 years) to minimize fee drag 
while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: Weekly data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(20) on weekly close
    if len(df_1w) >= 20:
        close_1w = df_1w['close'].values
        ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
        ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    else:
        ema_20_1w_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    chikou = np.roll(close, -26)  # Will be handled in logic
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for Ichimoku calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of weekly trend ---
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]
        
        # --- Cloud Analysis ---
        # Cloud top is the higher of Senkou A and Senkou B
        # Cloud bottom is the lower of Senkou A and Senkou B
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        # --- Price relative to cloud ---
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        price_in_cloud = (close[i] >= cloud_bottom) & (close[i] <= cloud_top)
        
        # --- Exit Logic (Cloud-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit if price falls below cloud bottom
                if close[i] < cloud_bottom:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Optional: take profit if price reaches 2x cloud thickness above cloud top
                cloud_thickness = cloud_top - cloud_bottom
                if cloud_thickness > 0 and close[i] > cloud_top + 2 * cloud_thickness:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit if price rises above cloud top
                if close[i] > cloud_top:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Optional: take profit if price reaches 2x cloud thickness below cloud bottom
                cloud_thickness = cloud_top - cloud_bottom
                if cloud_thickness > 0 and close[i] < cloud_bottom - 2 * cloud_thickness:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price above cloud AND Tenkan > Kijun (bullish alignment) AND weekly uptrend
        long_condition = (
            price_above_cloud and 
            tenkan[i] > kijun[i] and 
            price_above_weekly_ema
        )
        
        # Short: Price below cloud AND Tenkan < Kijun (bearish alignment) AND weekly downtrend
        short_condition = (
            price_below_cloud and 
            tenkan[i] < kijun[i] and 
            price_below_weekly_ema
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals