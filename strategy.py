#!/usr/bin/env python3
"""
Experiment #4945: 12h Camarilla Pivot + 1d Volume Spike + Choppiness Regime
HYPOTHESIS: On 12h timeframe, price reactions at 1d Camarilla pivot levels (L3, L4, H3, H4) with volume confirmation (>1.5x average) and choppiness regime filter (CHOP > 50 for mean reversion) capture high-probability reversals. Works in bull markets (bounces off support) and bear markets (rejections at resistance). Designed for 12-37 trades/year on 12h timeframe to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4945_12h_camarilla_pivot_1d_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Camarilla pivots, volume, and choppiness
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Camarilla Pivot Levels (L3, L4, H3, H4) ===
    if len(df_1d) >= 2:
        # Use previous day's OHLC for today's pivot levels (avoid look-ahead)
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_hl = prev_high - prev_low
        
        # Camarilla levels
        L3 = pivot - (range_hl * 1.1 / 4)
        L4 = pivot - (range_hl * 1.1 / 2)
        H3 = pivot + (range_hl * 1.1 / 4)
        H4 = pivot + (range_hl * 1.1 / 2)
    else:
        pivot = np.full(len(df_1d), np.nan)
        L3 = np.full(len(df_1d), np.nan)
        L4 = np.full(len(df_1d), np.nan)
        H3 = np.full(len(df_1d), np.nan)
        H4 = np.full(len(df_1d), np.nan)
    
    # Align HTF Camarilla levels to 12h timeframe
    if len(pivot) > 0:
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
        L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
        H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
        H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    else:
        pivot_aligned = np.full(n, np.nan)
        L3_aligned = np.full(n, np.nan)
        L4_aligned = np.full(n, np.nan)
        H3_aligned = np.full(n, np.nan)
        H4_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Volume MA (20-period) for spike detection ===
    if len(df_1d) >= 20:
        vol_ma_1d = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    else:
        vol_ma_1d = np.full(len(df_1d), np.nan)
    
    if len(vol_ma_1d) > 0:
        vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    else:
        vol_ma_1d_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Choppiness Index (CHOP) ===
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index: CHOP = 100 * log10(tr_sum / (hh_14 - ll_14)) / log10(14)
        # Avoid division by zero
        hl_range_14 = hh_14 - ll_14
        chop_1d = np.full(len(df_1d), np.nan)
        valid = (hl_range_14 > 0) & (~np.isnan(tr_sum))
        chop_1d[valid] = 100 * np.log10(tr_sum[valid] / hl_range_14[valid]) / np.log10(14)
    else:
        chop_1d = np.full(len(df_1d), np.nan)
    
    if len(chop_1d) > 0:
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        chop_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma_12h[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = max(20, 20, 14)  # Volume MA, Camarilla (need 2 days), CHOP warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(H4_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on opposite signal or volume drop ---
        if in_position:
            # Exit conditions: reverse signal or volume drops below average
            vol_exit = vol_ratio[i] < 1.0
            
            if position_side > 0:  # Long position
                # Exit if price reaches H3/H4 (resistance) or volume drops
                if (price >= H3_aligned[i]) or vol_exit:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit if price reaches L3/L4 (support) or volume drops
                if (price <= L3_aligned[i]) or vol_exit:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x average)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Choppiness filter: mean reversion regime (CHOP > 50)
        chop_filter = chop_1d_aligned[i] > 50
        
        # Mean reversion at Camarilla levels with volume and chop confirmation
        long_setup = (price <= L3_aligned[i]) and vol_confirm and chop_filter
        short_setup = (price >= H3_aligned[i]) and vol_confirm and chop_filter
        
        # Final entry conditions
        if long_setup:
            in_position = True
            position_side = 1
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals