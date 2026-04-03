#!/usr/bin/env python3
"""
Experiment #055: 6h Ichimoku Cloud + 1d ADX Trend Filter

HYPOTHESIS: Ichimoku cloud (TK cross + price vs cloud) provides high-probability
trend signals on 6h timeframe. 1d ADX > 25 filters for strong trending conditions
to avoid whipsaws in ranging markets. Volume confirmation ensures institutional
participation. Works in both bull and bear markets by following established trends
with strict filters to minimize false signals.
Target: 75-150 trades over 4 years (19-37/year) with discrete sizing (0.25).
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
    
    # === HTF: 1d data for ADX and volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range and Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Smoothing (Wilder's smoothing = EMA with alpha=1/period)
        def wilder_smoothing(data, period):
            result = np.zeros_like(data)
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        if len(tr) >= 14:
            atr = wilder_smoothing(tr, 14)
            if len(atr) >= 14:
                plus_di = 100 * wilder_smoothing(plus_dm, 14) / atr
                minus_di = 100 * wilder_smoothing(minus_dm, 14) / atr
                dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
                adx = wilder_smoothing(dx, 14)
                # Prepend zeros for alignment
                adx_full = np.zeros(len(close_1d))
                adx_full[14:] = adx[13:]  # Adjust for Wilder's smoothing offset
                adx_full[:14] = 20.0  # Neutral for warmup
                adx_aligned = align_htf_to_ltf(prices, df_1d, adx_full)
            else:
                adx_aligned = np.full(n, 20.0)
        else:
            adx_aligned = np.full(n, 20.0)
    else:
        adx_aligned = np.full(n, 20.0)
    
    # Calculate volume ratio on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 6h Ichimoku Cloud Components ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    if n >= period_tenkan:
        max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
        min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
        tenkan = (max_high_9 + min_low_9) / 2
    else:
        tenkan = np.full(n, np.nan)
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    if n >= period_kijun:
        max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
        min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
        kijun = (max_high_26 + min_low_26) / 2
    else:
        kijun = np.full(n, np.nan)
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    if n >= period_senkou_b:
        max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
        min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
        senkou_b = (max_high_52 + min_low_52) / 2
    else:
        senkou_b = np.full(n, np.nan)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    chikou = np.roll(close, -26)  # Will be handled in logic
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i])):
            signals[i] = 0.0
            continue
        
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Determine conditions
        # Price above/below cloud
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK Cross
        tk_cross_bull = tenkan[i] > kijun[i]
        tk_cross_bear = tenkan[i] < kijun[i]
        
        # ADX trend strength (1d)
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation (1d)
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # Long conditions: bullish TK cross + price above cloud + strong trend + volume
        if tk_cross_bull and price_above_cloud and strong_trend and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        
        # Short conditions: bearish TK cross + price below cloud + strong trend + volume
        elif tk_cross_bear and price_below_cloud and strong_trend and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        
        # No signal
        else:
            signals[i] = 0.0
    
    return signals