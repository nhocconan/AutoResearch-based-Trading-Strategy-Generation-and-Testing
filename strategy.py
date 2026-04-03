#!/usr/bin/env python3
"""
Experiment #355: 6h Weekly Camarilla Pivot Breakout with Volume Spike

HYPOTHESIS: Weekly Camarilla pivot levels (R4/S4) act as strong support/resistance.
Breakouts above R4 or below S4 with volume spike (>2x 20-period average) continue
in the direction of the breakout with momentum. In ranging markets (price between
R3/S3), mean reversion to the weekly pivot point (PP) occurs. This combines
breakout and mean reversion logic adapted to weekly structure on 6h timeframe,
targeting 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_355_6h_weekly_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly Camarilla pivots ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels
    def calculate_camarilla(h, l, c):
        """Calculate Camarilla pivot levels: R4, R3, R2, R1, PP, S1, S2, S3, S4"""
        range_ = h - l
        pp = (h + l + c) / 3.0
        r4 = c + range_ * 1.1 / 2.0
        r3 = c + range_ * 1.1 / 4.0
        r2 = c + range_ * 1.1 / 6.0
        r1 = c + range_ * 1.1 / 12.0
        s1 = c - range_ * 1.1 / 12.0
        s2 = c - range_ * 1.1 / 6.0
        s3 = c - range_ * 1.1 / 4.0
        s4 = c - range_ * 1.1 / 2.0
        return r4, r3, r2, r1, pp, s1, s2, s3, s4
    
    # Calculate for each 1w bar
    r4_1w = np.full(len(df_1w), np.nan)
    r3_1w = np.full(len(df_1w), np.nan)
    s3_1w = np.full(len(df_1w), np.nan)
    s4_1w = np.full(len(df_1w), np.nan)
    pp_1w = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        r4, r3, r2, r1, pp, s1, s2, s3, s4 = calculate_camarilla(
            df_1w['high'].iloc[i], 
            df_1w['low'].iloc[i], 
            df_1w['close'].iloc[i]
        )
        r4_1w[i] = r4
        r3_1w[i] = r3
        s3_1w[i] = s3
        s4_1w[i] = s4
        pp_1w[i] = pp
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
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
    
    warmup = 100  # Warmup for stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(pp_1w_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Price Levels ---
        price = close[i]
        r4 = r4_1w_aligned[i]
        r3 = r3_1w_aligned[i]
        s3 = s3_1w_aligned[i]
        s4 = s4_1w_aligned[i]
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Exit on mean reversion to weekly PP in ranging markets
            # Ranging defined as price between R3 and S3
            if r3 > price > s3 and abs(price - pp) < 0.5 * atr_14[i]:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long breakout: Price > R4 + volume spike
        long_breakout = (price > r4) and volume_spike
        
        # Short breakout: Price < S4 + volume spike
        short_breakout = (price < s4) and volume_spike
        
        # Long mean reversion: Price < R3 + ranging (between R3/S3) + below PP
        long_mr = (price < r3) and (price > s3) and (price < pp)
        
        # Short mean reversion: Price > S3 + ranging (between R3/S3) + above PP
        short_mr = (price > s3) and (price < r3) and (price > pp)
        
        if long_breakout or long_mr:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_breakout or short_mr:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals