#!/usr/bin/env python3
"""
Experiment #1908: 6h Ichimoku Cloud + 1d ADX Trend Filter + Volume Spike
HYPOTHESIS: Ichimoku cloud on 6h provides dynamic support/resistance, while 1d ADX > 25 filters for trending markets only.
Entry: Tenkan-Kijun cross + price above/below cloud + volume > 2x average, aligned with 1d ADX trend direction.
Exit: Opposite Tenkan-Kijun cross or price penetrates cloud by 50%.
Works in bull/bear by only trading strong trends (ADX filter) and using cloud as dynamic S/R.
Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1908_6h_ichimoku_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    close = prices["close"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    period = 14
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    tr_period = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    dm_plus_period = pd.Series(dm_plus).rolling(window=period, min_periods=period).sum().values
    dm_minus_period = pd.Series(dm_minus).rolling(window=period, min_periods=period).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    # 1d trend: ADX > 25 = trending, direction from DI+ > DI-
    trend_1d = np.where((adx > 25) & (di_plus > di_minus), 1, 
                        np.where((adx > 25) & (di_minus > di_plus), -1, 0))
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Ichimoku Components ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9 = 9
    high9 = pd.Series(high).rolling(window=period9, min_periods=period9).max().values
    low9 = pd.Series(low).rolling(window=period9, min_periods=period9).min().values
    tenkan = (high9 + low9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26 = 26
    high26 = pd.Series(high).rolling(window=period26, min_periods=period26).max().values
    low26 = pd.Series(low).rolling(window=period26, min_periods=period26).min().values
    kijun = (high26 + low26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52 = 52
    high52 = pd.Series(high).rolling(window=period52, min_periods=period52).max().values
    low52 = pd.Series(low).rolling(window=period52, min_periods=period52).min().values
    senkou_b = ((high52 + low52) / 2.0)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # For signal generation, we compare current price with Senkou spans
    
    # Cloud boundaries (Senkou A and B)
    # For bullish cloud: Senkou A > Senkou B
    # For bearish cloud: Senkou A < Senkou B
    # We need current cloud values (not shifted) for price comparison
    # Ichimoku cloud at time t uses Senkou A/B calculated t-26 periods ago
    # So current cloud = Senkou A/B from 26 periods ago
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # === 6h Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for Ichimoku (52) and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            # Calculate cloud penetration (how far price is inside cloud)
            if cloud_top[i] > cloud_bottom[i]:  # valid cloud
                cloud_middle = (cloud_top[i] + cloud_bottom[i]) / 2.0
                if position_side > 0:  # Long
                    # Exit if price falls below cloud middle (50% penetration)
                    if price < cloud_middle:
                        exit_signal = True
                else:  # Short
                    # Exit if price rises above cloud middle
                    if price > cloud_middle:
                        exit_signal = True
            
            # Alternative exit: Tenkan-Kijun cross in opposite direction
            if not exit_signal and i > 0:
                tk_cross = tenkan[i] - kijun[i]
                tk_cross_prev = tenkan[i-1] - kijun[i-1]
                if position_side > 0 and tk_cross < 0 and tk_cross_prev > 0:
                    exit_signal = True  # bearish cross
                elif position_side < 0 and tk_cross > 0 and tk_cross_prev < 0:
                    exit_signal = True  # bullish cross
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike and trend_bias != 0:
            # Tenkan-Kijun cross
            tk_cross = tenkan[i] - kijun[i]
            tk_cross_prev = tenkan[i-1] - kijun[i-1] if i > 0 else 0
            
            # Bullish TK cross: Tenkan crosses above Kijun
            bullish_tk = tk_cross > 0 and tk_cross_prev <= 0
            # Bearish TK cross: Tenkan crosses below Kijun
            bearish_tk = tk_cross < 0 and tk_cross_prev >= 0
            
            # Price relative to cloud
            price_above_cloud = price > cloud_top[i]
            price_below_cloud = price < cloud_bottom[i]
            
            if bullish_tk and price_above_cloud and trend_bias > 0:
                # Long: bullish TK cross, price above cloud, 1d trend up
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif bearish_tk and price_below_cloud and trend_bias < 0:
                # Short: bearish TK cross, price below cloud, 1d trend down
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals