#!/usr/bin/env python3
"""
Experiment #039: 6h Bollinger Band Squeeze + 12h Volume Spike + 1d ADX Trend Filter

HYPOTHESIS: Bollinger Band squeeze (low volatility) on 6h timeframe identifies periods of consolidation 
that precede explosive moves. Combined with 12h volume spike confirmation (institutional participation) 
and 1d ADX > 25 (trending regime filter), this strategy captures high-probability breakouts in both 
bull and bear markets. The squeeze filter reduces false breakouts, volume confirms legitimacy, and 
ADX ensures we only trade when trends are strong enough to sustain moves. Targets 15-25 trades/year 
on 6h timeframe (60-100 total over 4 years) to minimize fee drag while capturing explosive moves 
after low volatility periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_bb_squeeze_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d data
    if len(df_1d) >= 30:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                           np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                            np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr_1d = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr_1d
        di_minus = 100 * dm_minus_smooth / atr_1d
        
        # ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        dx = np.where((di_plus + di_minus) == 0, 0, dx)
        adx_1d = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    else:
        adx_1d_aligned = np.full(n, 20.0)  # Default to non-trending
    
    # === 6h Indicators ===
    # Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    if n >= bb_period:
        bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
        bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
        bb_upper = bb_ma + bb_std * bb_stddev
        bb_lower = bb_ma - bb_std * bb_stddev
        bb_width = (bb_upper - bb_lower) / bb_ma  # Normalized width
        
        # Bollinger Band Squeeze: width below 20-period average of width
        bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
        bb_squeeze = bb_width < bb_width_ma  # True when in squeeze (low volatility)
    else:
        bb_ma = np.full(n, np.nan)
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
        bb_squeeze = np.full(n, False)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, bb_period + 20)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(bb_ma[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filters ---
        # 1. Bollinger Band Squeeze (low volatility setup)
        in_squeeze = bb_squeeze[i]
        
        # 2. Volume Confirmation: Require volume spike (> 2.0x average) on 12h
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
        # 3. ADX Trend Filter: Only trade when ADX > 25 (trending regime)
        strong_trend = adx_1d_aligned[i] > 25.0
        
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
                # Take profit at upper BB (mean reversion) or strong continuation
                if close[i] >= bb_upper[i]:
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
                # Take profit at lower BB (mean reversion) or strong continuation
                if close[i] <= bb_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require: squeeze setup + volume spike + strong trend
        setup_conditions = in_squeeze and volume_spike and strong_trend
        
        if setup_conditions:
            # Long: Price breaks above upper BB with volume
            if close[i] > bb_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short: Price breaks below lower BB with volume
            elif close[i] < bb_lower[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals