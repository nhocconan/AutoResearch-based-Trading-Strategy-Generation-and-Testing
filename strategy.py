#!/usr/bin/env python3
"""
Experiment #375: 6h Weekly Pivot + Volume Confirmation + ATR Stop

HYPOTHESIS: Weekly pivot points (R1/S1, R2/S2) provide strong institutional support/resistance. 
Price approaching weekly S1/R1 with volume confirmation offers high-probability mean reversion 
entries, while breaks of weekly R2/S2 with volume indicate continuation. The 6h timeframe 
captures these reactions with sufficient precision. Volume confirms institutional participation. 
ATR-based stoploss manages risk. Targets 12-37 trades/year (50-150 total over 4 years) to 
minimize fee drag while capturing high-probability reactions at weekly pivot levels in both 
bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_vol_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: Weekly data for pivot points (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    if len(df_1w) >= 1:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # Weekly pivot calculations
        pivot_1w = (high_1w + low_1w + close_1w) / 3
        range_1w = high_1w - low_1w
        r1_1w = 2 * pivot_1w - low_1w
        s1_1w = 2 * pivot_1w - high_1w
        r2_1w = pivot_1w + range_1w
        s2_1w = pivot_1w - range_1w
        
        # Align to 6h timeframe (shifted by 1 week for completed bars only)
        pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
        r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
        s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
        r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
        s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    else:
        pivot_1w_aligned = np.full(n, np.nan)
        r1_1w_aligned = np.full(n, np.nan)
        s1_1w_aligned = np.full(n, np.nan)
        r2_1w_aligned = np.full(n, np.nan)
        s2_1w_aligned = np.full(n, np.nan)
    
    # === HTF: Daily data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 6h Indicators: ATR for stoploss ===
    # Calculate ATR(14) on 6h
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at weekly R2 (strong resistance)
                if close[i] >= r2_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at weekly S2 (strong support)
                if close[i] <= s2_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price at S1 (mean reversion) OR break above R2 with volume
        long_condition = (
            (close[i] <= s1_1w_aligned[i] * 1.002) or  # S1 mean reversion (0.2% buffer)
            (close[i] > r2_1w_aligned[i] and vol_ratio_1d_aligned[i] > 2.0)  # Breakout with volume spike
        )
        
        # Short: Price at R1 (mean reversion) OR break below S2 with volume
        short_condition = (
            (close[i] >= r1_1w_aligned[i] * 0.998) or  # R1 mean reversion (0.2% buffer)
            (close[i] < s2_1w_aligned[i] and vol_ratio_1d_aligned[i] > 2.0)  # Breakdown with volume spike
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