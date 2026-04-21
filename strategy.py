#!/usr/bin/env python3
"""
6h_IWM_Regime_Adaptive_Momentum
Hypothesis: Adaptive strategy using 6h Internals (Advance-Decline Line proxy via volume-weighted RSI) combined with 12h Ichimoku cloud filter and 1d ADX regime detection.
In bull/bear regimes (ADX>25): trade with Ichimoku TK cross + cloud color.
In range regimes (ADX<20): mean revert at 6h volume-weighted RSI extremes.
Volume-weighted RSI acts as internal market strength proxy, working across crypto cycles.
Designed for low trade frequency (<25 trades/year) via regime filters and strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 6h Volume-Weighted RSI (proxy for internal strength) ===
    close = prices['close'].values
    volume = prices['volume'].values
    # Typical price
    tp = (prices['high'].values + prices['low'].values + close) / 3.0
    # Volume-weighted typical price change
    vwtp = tp * volume
    # Calculate changes
    delta = np.diff(vwtp, prepend=vwtp[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # Smoothed averages
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    # === 12h Ichimoku Cloud ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2.0)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Align Ichimoku components to 6h
    tenkan_6h = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_12h, senkou_a, additional_delay_bars=26)
    senkou_b_6h = align_htf_to_ltf(prices, df_12h, senkou_b, additional_delay_bars=26)
    # Kumo (cloud) top and bottom
    kumotop_6h = np.maximum(senkou_a_6h, senkou_b_6h)
    kumobottom_6h = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # === 1d ADX for Regime Detection ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff() * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(vw_rsi[i]) or np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) 
            or np.isnan(kumotop_6h[i]) or np.isnan(kumobottom_6h[i]) or np.isnan(adx_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        adx_val = adx_6h[i]
        
        if position == 0:
            # Regime-based entry
            if adx_val > 25:  # Trending regime
                # Ichimoku TK cross with cloud filter
                tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
                tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
                price_above_kumo = price > kumotop_6h[i]
                price_below_kumo = price < kumobottom_6h[i]
                
                if tk_cross_up and price_above_kumo:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif tk_cross_down and price_below_kumo:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    
            elif adx_val < 20:  # Range regime
                # Mean reversion at VW-RSI extremes
                if vw_rsi[i] < 30 and price > close[i-1]:  # Oversold + bullish price action
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif vw_rsi[i] > 70 and price < close[i-1]:  # Overbought + bearish price action
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Exit conditions
            if adx_val > 25:  # Trending regime exit
                tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
                price_below_kumo = price < kumobottom_6h[i]
                if tk_cross_down or price_below_kumo:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Range regime exit
                if vw_rsi[i] > 50:  # Return to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:
            # Exit conditions
            if adx_val > 25:  # Trending regime exit
                tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
                price_above_kumo = price > kumotop_6h[i]
                if tk_cross_up or price_above_kumo:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Range regime exit
                if vw_rsi[i] < 50:  # Return to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_IWM_Regime_Adaptive_Momentum"
timeframe = "6h"
leverage = 1.0