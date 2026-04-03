#!/usr/bin/env python3
"""
Experiment #407: 6h Ichimoku Cloud + 1d ADX Trend + Volume Confirmation

HYPOTHESIS: Ichimoku Cloud (TK cross + price vs cloud) on 6h timeframe, 
combined with 1d ADX > 25 for trend strength and volume confirmation (> 1.5x average), 
creates a robust strategy that works in both bull and bear markets. 
Ichimoku provides dynamic support/resistance (cloud) and momentum (TK cross), 
while 1d ADX ensures we only trade strong trends, reducing whipsaw in ranging markets. 
Volume confirms institutional participation. Targets 12-37 trades/year on 6h timeframe 
(50-150 total over 4 years) to minimize fee drag while capturing high-probability 
trend continuation signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, 0.0)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 6h Indicators: Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    if n >= period_tenkan:
        tenkan_sen = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                      pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
        tenkan_sen = tenkan_sen.values
    else:
        tenkan_sen = np.full(n, np.nan)
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    if n >= period_kijun:
        kijun_sen = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                     pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
        kijun_sen = kijun_sen.values
    else:
        kijun_sen = np.full(n, np.nan)
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
    if not np.all(np.isnan(tenkan_sen)) and not np.all(np.isnan(kijun_sen)):
        senkou_a = ((tenkan_sen + kijun_sen) / 2)
    else:
        senkou_a = np.full(n, np.nan)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    if n >= period_senkou_b:
        senkou_b = (pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                    pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
        senkou_b = senkou_b.values
    else:
        senkou_b = np.full(n, np.nan)
    
    # Current cloud boundaries (shifted back by 26 to align with current price)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Upper and lower cloud boundaries
    upper_cloud = np.maximum(senkou_a_shifted, senkou_b_shifted)
    lower_cloud = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in strong trending markets (ADX > 25) ---
        strong_trend = adx_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Ichimoku Conditions ---
        price_above_cloud = close[i] > upper_cloud[i]
        price_below_cloud = close[i] < lower_cloud[i]
        tk_cross_bull = tenkan_sen[i] > kijun_sen[i]
        tk_cross_bear = tenkan_sen[i] < kijun_sen[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit when price crosses below cloud (trend change)
                if close[i] < upper_cloud[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit when price crosses above cloud (trend change)
                if close[i] > lower_cloud[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price above cloud + TK cross bullish + strong trend + volume
        long_condition = (
            price_above_cloud and 
            tk_cross_bull and 
            strong_trend and 
            volume_spike
        )
        
        # Short: Price below cloud + TK cross bearish + strong trend + volume
        short_condition = (
            price_below_cloud and 
            tk_cross_bear and 
            strong_trend and 
            volume_spike
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