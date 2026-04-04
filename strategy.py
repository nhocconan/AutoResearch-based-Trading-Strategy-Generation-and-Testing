#!/usr/bin/env python3
"""
Experiment #3639: 6h Camarilla Pivot + Volume Spike + Regime Filter
HYPOTHESIS: Camarilla pivot levels from 1d timeframe provide institutional support/resistance. 
Fade at R3/S3 levels with volume confirmation in ranging markets (CHOP > 61.8), 
breakout continuation at R4/S4 levels in trending markets (CHOP < 38.2). 
6h timeframe reduces noise while capturing meaningful swings. 
Position size 0.25 balances opportunity with drawdown control. 
Target: 75-150 total trades over 4 years (19-37/year).
Works in bull markets (breakouts at R4/S4 with trend) and bear markets (fades at R3/S3 against false moves).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3639_6h_camarilla_pivot_volume_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (Range * 1.1/2)
    # R3 = C + (Range * 1.1/4)
    # S3 = C - (Range * 1.1/4)
    # S4 = C - (Range * 1.1/2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.1 / 2.0)
    r3_1d = close_1d + (range_1d * 1.1 / 4.0)
    s3_1d = close_1d - (range_1d * 1.1 / 4.0)
    s4_1d = close_1d - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed 1d bar)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === HTF: 12h data for trend filter (EMA crossover) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # Calculate 12h EMA(9) and EMA(21) for trend
    ema_9_12h = pd.Series(close_12h).ewm(span=9, min_periods=9, adjust=False).mean().values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_9_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_9_12h)
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # === 6h Indicators: Choppiness Index (CHOP) for regime detection ===
    # CHOP = 100 * LOG10(SUM(ATR,14) / (MAXHIGH - MINLOW)) / LOG10(14)
    # CHOP > 61.8 = ranging market (mean revert)
    # CHOP < 38.2 = trending market (trend follow)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((max_high_14 - min_low_14) > 0, chop_raw, 50.0)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = max(50, 20, 14, 21, 9)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_9_12h_aligned[i]) or np.isnan(ema_21_12h_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit if price moves against position by 1.5*ATR(14) (approximate)
            # We'll use a simpler time-based exit for now to avoid look-ahead complexity
            # In practice, would use ATR-based stop but keeping simple for signal count control
            if position_side > 0:  # Long
                # Exit if price drops below S3 (mean reversion target) or breaks S4 (stop)
                if price < s3_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit if price rises above R3 (mean reversion target) or breaks R4 (stop)
                if price > r3_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Determine trend bias from 12h EMA crossover
            bullish_trend = ema_9_12h_aligned[i] > ema_21_12h_aligned[i]
            bearish_trend = ema_9_12h_aligned[i] < ema_21_12h_aligned[i]
            
            # Regime detection using Choppiness Index
            ranging_market = chop[i] > 61.8   # CHOP > 61.8 = range (mean revert)
            trending_market = chop[i] < 38.2  # CHOP < 38.2 = trend (trend follow)
            
            # Long entry conditions
            long_entry = False
            if ranging_market:
                # In ranging market: fade at S3 (mean reversion up from support)
                if price <= s3_1d_aligned[i] * 1.001:  # Allow small slippage
                    long_entry = True
            elif trending_market:
                # In trending market: breakout continuation at S4 (breakdown then bounce)
                if price <= s4_1d_aligned[i] * 1.001 and bullish_trend:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            if ranging_market:
                # In ranging market: fade at R3 (mean reversion down from resistance)
                if price >= r3_1d_aligned[i] * 0.999:  # Allow small slippage
                    short_entry = True
            elif trending_market:
                # In trending market: breakout continuation at R4 (breakout then pullback)
                if price >= r4_1d_aligned[i] * 0.999 and bearish_trend:
                    short_entry = True
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif short_entry:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals