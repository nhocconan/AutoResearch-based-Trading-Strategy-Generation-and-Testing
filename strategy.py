#!/usr/bin/env python3
"""
Experiment #3431: 6h Camarilla Pivot + 1d Volume Spike + ADX Regime Filter
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) combined with 1d volume confirmation and ADX regime filter (ADX>25 for trending, ADX<20 for ranging) captures high-probability entries in both bull and bear markets. Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3431_6h_camarilla_pivot_1d_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume and ADX regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume MA(20) for spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones(len(close_1d))
    vol_ratio_1d[20:] = volume_1d[20:] / vol_ma_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate 1d ADX(14) for regime filter
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values / atr
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Camarilla pivot levels from previous 1d bar ===
    # Camarilla levels calculated from previous day's OHLC
    # We use the 1d data shifted by 1 to avoid look-ahead (previous completed day)
    close_1d_prev = np.concatenate([[np.nan], close_1d[:-1]])
    high_1d_prev = np.concatenate([[np.nan], high_1d[:-1]])
    low_1d_prev = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Calculate Camarilla levels for each 1d bar (based on previous day)
    camarilla_range = high_1d_prev - low_1d_prev
    camarilla_r3 = close_1d_prev + camarilla_range * 1.1 / 4
    camarilla_s3 = close_1d_prev - camarilla_range * 1.1 / 4
    camarilla_r4 = close_1d_prev + camarilla_range * 1.1 / 2
    camarilla_s4 = close_1d_prev - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    
    warmup = max(20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Fixed stoploss at 2.0 * ATR from entry
            if position_side > 0:  # Long
                if price < stoploss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > stoploss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d volume spike (> 1.5x average) for confirmation
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # Regime filter: ADX > 25 = trending (breakout), ADX < 20 = ranging (mean reversion)
        adx_val = adx_1d_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if volume_spike:
            if is_trending:
                # Trending market: breakout continuation at R4/S4
                if price > r4_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    stoploss_price = entry_price - 2.0 * atr[i]
                    signals[i] = SIZE
                elif price < s4_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    stoploss_price = entry_price + 2.0 * atr[i]
                    signals[i] = -SIZE
            elif is_ranging:
                # Ranging market: mean reversion at R3/S3
                if price < r3_aligned[i] and price > s3_aligned[i]:
                    # Look for reversal signals near pivot levels
                    if price <= s3_aligned[i] * 1.005:  # Near S3, potential long
                        in_position = True
                        position_side = 1
                        entry_price = close[i]
                        stoploss_price = entry_price - 2.0 * atr[i]
                        signals[i] = SIZE
                    elif price >= r3_aligned[i] * 0.995:  # Near R3, potential short
                        in_position = True
                        position_side = -1
                        entry_price = close[i]
                        stoploss_price = entry_price + 2.0 * atr[i]
                        signals[i] = -SIZE
    
    return signals