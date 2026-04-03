#!/usr/bin/env python3
"""
Experiment #096: 12h Camarilla pivot levels + volume confirmation + choppiness regime

HYPOTHESIS: On 12h timeframe, price reversals at Camarilla pivot levels (S3, S4, R3, R4) derived
from prior 1d session, confirmed by 1h volume spike and choppiness regime filter (CHOP > 61.8 = range),
capture mean-reversion moves in both bull and bear markets. The 12h timeframe reduces noise while
Camarilla levels provide high-probability reversal zones. Volume spike confirms participation, chop
filter ensures ranging conditions. Targets 12-37 trades/year (50-150 total over 4 years) to minimize
fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior 1d bar (H, L, C)
    if len(df_1d) >= 1:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        # Camarilla formula: Range = H - L
        # S1 = C - (Range * 1.1/12), S2 = C - (Range * 1.1/6), S3 = C - (Range * 1.1/4)
        # S4 = C - (Range * 1.1/2), R4 = C + (Range * 1.1/2), R3 = C + (Range * 1.1/4)
        # R2 = C + (Range * 1.1/6), R1 = C + (Range * 1.1/12)
        range_1d = high_1d - low_1d
        camarilla_s4 = close_1d - (range_1d * 1.1 / 2)
        camarilla_s3 = close_1d - (range_1d * 1.1 / 4)
        camarilla_r3 = close_1d + (range_1d * 1.1 / 4)
        camarilla_r4 = close_1d + (range_1d * 1.1 / 2)
        # Align to 12h timeframe (shift by 1 to use completed 1d bar)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    else:
        camarilla_s4_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_r4_aligned = np.full(n, np.nan)
    
    # === MTF: 1h data for volume confirmation (Call ONCE before loop) ===
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate volume ratio (current vs 20-period average) on 1h
    if len(df_1h) >= 20:
        vol_1h = df_1h['volume'].values
        vol_ma_20 = pd.Series(vol_1h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1h = np.zeros(len(vol_1h))
        vol_ratio_1h[20:] = vol_1h[20:] / vol_ma_20[20:]
        vol_ratio_1h[:20] = 1.0  # Neutral for warmup
        vol_ratio_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_ratio_1h)
    else:
        vol_ratio_1h_aligned = np.full(n, 1.0)
    
    # === 12h Indicators ===
    # Choppiness Index (CHOP) - range detection
    def choppiness_index(high, low, close, period=14):
        """Calculate Choppiness Index: higher values = more choppy/ranging"""
        atr_sum = np.zeros(len(close))
        true_range = np.zeros(len(close))
        for i in range(len(close)):
            if i == 0:
                true_range[i] = high[i] - low[i]
            else:
                true_range[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
            if i < period:
                atr_sum[i] = np.nan
            else:
                start = i - period + 1
                atr_sum[i] = np.sum(true_range[start:i+1])
        
        # Calculate CHOP: 100 * log10(atr_sum / (period * true_range)) / log10(period)
        chop = np.full(len(close), np.nan)
        for i in range(period-1, len(close)):
            if atr_sum[i] > 0 and true_range[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (period * true_range[i])) / np.log10(period)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(vol_ratio_1h_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Choppiness Index > 61.8 = ranging market (mean revert) ---
        ranging_market = chop[i] > 61.8
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) on 1h ---
        volume_spike = vol_ratio_1h_aligned[i] > 1.5
        
        # --- Price levels ---
        price = close[i]
        s4 = camarilla_s4_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r3 = camarilla_r3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        
        # --- Exit Logic (Mean reversion: exit at opposite level or midpoint) ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit at S3 (profit target) or midpoint (S3+S4)/2
                exit_level = (s3 + s4) / 2
                if price <= exit_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit at R3 (profit target) or midpoint (R3+R4)/2
                exit_level = (r3 + r4) / 2
                if price >= exit_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price touches/below S4 with volume and ranging market
        long_condition = (
            price <= s4 and 
            volume_spike and 
            ranging_market
        )
        
        # Short: Price touches/above R4 with volume and ranging market
        short_condition = (
            price >= r4 and 
            volume_spike and 
            ranging_market
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = price
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = price
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals