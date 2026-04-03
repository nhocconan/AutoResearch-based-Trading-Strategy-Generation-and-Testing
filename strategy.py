#!/usr/bin/env python3
"""
Experiment #227: 6h Weekly Pivot + Volume Spike Regime Adaptive

HYPOTHESIS: Weekly pivot levels (R3/S3) act as strong support/resistance. In ranging markets (weekly ADX < 20), 
price reverts to weekly pivot point after touching R3/S3 with volume confirmation. In trending markets (weekly ADX > 25),
breakouts above R4 or below S4 with volume spike continue the trend. This dual-regime approach adapts to 
BTC/ETH/SOL behavior across bull/bear cycles. 6h timeframe targets 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_227_6h_weekly_pivot_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Weekly Pivots and ADX regime (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Weekly Pivot levels (Standard Floor Pivots)
    def calculate_weekly_pivots(h, l, c):
        """Calculate standard weekly pivot levels: R3, R2, R1, PP, S1, S2, S3"""
        pp = (h + l + c) / 3.0
        r1 = 2 * pp - l
        s1 = 2 * pp - h
        r2 = pp + (h - l)
        s2 = pp - (h - l)
        r3 = h + 2 * (pp - l)
        s3 = l - 2 * (h - pp)
        return r3, r2, r1, pp, s1, s2, s3
    
    # Calculate for each weekly bar
    r3_1w = np.full(len(df_1w), np.nan)
    r2_1w = np.full(len(df_1w), np.nan)
    r1_1w = np.full(len(df_1w), np.nan)
    pp_1w = np.full(len(df_1w), np.nan)
    s1_1w = np.full(len(df_1w), np.nan)
    s2_1w = np.full(len(df_1w), np.nan)
    s3_1w = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        r3, r2, r1, pp, s1, s2, s3 = calculate_weekly_pivots(
            df_1w['high'].iloc[i], 
            df_1w['low'].iloc[i], 
            df_1w['close'].iloc[i]
        )
        r3_1w[i] = r3
        r2_1w[i] = r2
        r1_1w[i] = r1
        pp_1w[i] = pp
        s1_1w[i] = s1
        s2_1w[i] = s2
        s3_1w[i] = s3
    
    # Align Weekly Pivot levels to 6h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Calculate 1w ADX for regime detection
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX (Average Directional Index)"""
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_1w = calculate_adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for weekly indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(adx_1w_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Regime Filter: ADX < 20 = ranging, ADX > 25 = trending ---
        is_ranging = adx_1w_aligned[i] < 20
        is_trending = adx_1w_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Price Levels ---
        price = close[i]
        r3 = r3_1w_aligned[i]
        s3 = s3_1w_aligned[i]
        pp = pp_1w_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on mean reversion to weekly PP in ranging markets
                if is_ranging and abs(price - pp) < 0.5 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on mean reversion to weekly PP in ranging markets
                if is_ranging and abs(price - pp) < 0.5 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long reversion: Price < S3 + volume spike + ranging + below PP
        long_reversion = (price < s3) and volume_spike and is_ranging and (price < pp)
        
        # Short reversion: Price > R3 + volume spike + ranging + above PP
        short_reversion = (price > r3) and volume_spike and is_ranging and (price > pp)
        
        # Long breakout: Price > R3 + volume spike + trending + above PP
        long_breakout = (price > r3) and volume_spike and is_trending and (price > pp)
        
        # Short breakout: Price < S3 + volume spike + trending + below PP
        short_breakout = (price < s3) and volume_spike and is_trending and (price < pp)
        
        if long_reversion or long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_reversion or short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals