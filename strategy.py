#!/usr/bin/env python3
"""
Experiment #287: 6h Camarilla Pivot + 1d Volume Spike + 1w Trend Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
filtered by 1d volume spikes (>2.0x average) and 1w EMA trend (EMA50) produce high-probability 
trades. In ranging markets, price reverts to mean at R3/S3. In trending markets, breakouts 
at R4/S4 continue with the weekly trend. Volume spike confirms institutional participation. 
Weekly EMA50 filter avoids counter-trend trades. Targets 50-150 trades over 4 years (12-37/year) 
to minimize fee drag while capturing meaningful moves. Works in bull (breakouts with volume) 
and bear (mean reversion at extremes) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_287_6h_camarilla_1d_volume_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and volume MA (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d
    def calculate_camarilla(h, l, c):
        """Calculate Camarilla pivot levels: R4, R3, R2, R1, PP, S1, S2, S3, S4"""
        rng = h - l
        pp = (h + l + c) / 3.0
        r4 = c + rng * 1.1 / 2.0
        r3 = c + rng * 1.1 / 4.0
        r2 = c + rng * 1.1 / 6.0
        r1 = c + rng * 1.1 / 12.0
        s1 = c - rng * 1.1 / 12.0
        s2 = c - rng * 1.1 / 6.0
        s3 = c - rng * 1.1 / 4.0
        s4 = c - rng * 1.1 / 2.0
        return r4, r3, r2, r1, pp, s1, s2, s3, s4
    
    # Calculate pivots for each 1d bar
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        c = df_1d['close'].iloc[i]
        r4, r3, _, _, _, _, _, s3, s4 = calculate_camarilla(h, l, c)
        camarilla_r4[i] = r4
        camarilla_r3[i] = r3
        camarilla_s3[i] = s3
        camarilla_s4[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d volume MA(20) for spike detection
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_6h = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === HTF: 1w data for EMA50 trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w data
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20_6h_direct = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Ensure enough data for HTF indicators, ATR, and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r3_6h[i]) or np.isnan(camarilla_s3_6h[i]) or 
            np.isnan(camarilla_r4_6h[i]) or np.isnan(camarilla_s4_6h[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ma_20_6h[i]) or np.isnan(vol_ma_20_6h_direct[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) from both timeframes ---
        volume_spike_1d = volume[i] > (2.0 * vol_ma_20_6h[i])
        volume_spike_6h = volume[i] > (2.0 * vol_ma_20_6h_direct[i])
        volume_spike = volume_spike_1d and volume_spike_6h
        
        # --- 1w EMA Trend Filter ---
        price_above_weekly_ema = close[i] > ema_50_6h[i]
        price_below_weekly_ema = close[i] < ema_50_6h[i]
        
        # --- Camarilla Conditions ---
        at_r3 = abs(close[i] - camarilla_r3_6h[i]) / camarilla_r3_6h[i] < 0.002  # Within 0.2% of R3
        at_s3 = abs(close[i] - camarilla_s3_6h[i]) / camarilla_s3_6h[i] < 0.002  # Within 0.2% of S3
        breakout_r4 = close[i] > camarilla_r4_6h[i]
        breakdown_s4 = close[i] < camarilla_s4_6h[i]
        
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
                # Take profit at opposite Camarilla level
                if position_side == 1 and close[i] < camarilla_s3_6h[i]:  # Long taking profit at S3
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
                # Take profit at opposite Camarilla level
                if position_side == -1 and close[i] > camarilla_r3_6h[i]:  # Short taking profit at R3
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
        # Long mean reversion at S3: price at S3 + volume spike + above weekly EMA
        long_mr = at_s3 and volume_spike and price_above_weekly_ema
        
        # Short mean reversion at R3: price at R3 + volume spike + below weekly EMA
        short_mr = at_r3 and volume_spike and price_below_weekly_ema
        
        # Long breakout continuation at R4: break above R4 + volume spike + above weekly EMA
        long_break = breakout_r4 and volume_spike and price_above_weekly_ema
        
        # Short breakdown continuation at S4: break below S4 + volume spike + below weekly EMA
        short_break = breakdown_s4 and volume_spike and price_below_weekly_ema
        
        if long_mr or long_break:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_mr or short_break:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals