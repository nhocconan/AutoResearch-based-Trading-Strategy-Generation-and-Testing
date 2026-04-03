#!/usr/bin/env python3
"""
Experiment #315: 6h Weekly Pivot + 1d ATR Breakout + Volume Confirmation

HYPOTHESIS: Weekly pivot points (calculated from prior week's OHLC) provide significant support/resistance levels.
Price breaking above/below these levels with 1d ATR expansion and volume confirmation indicates institutional participation.
In bull markets: break above weekly R1 with volume = continuation long.
In bear markets: break below weekly S1 with volume = continuation short.
The 6h timeframe captures the breakout with delay for confirmation, reducing false signals.
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_atr_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ATR (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR(14) on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i], 
                           abs(high_1d[i] - close_1d[i-1]),
                           abs(low_1d[i] - close_1d[i-1]))
        
        atr_14_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    else:
        atr_14_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for weekly pivot (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels (R1, S1, R2, S2) from prior week OHLC
    weekly_r1 = np.full(n, np.nan)
    weekly_s1 = np.full(n, np.nan)
    weekly_r2 = np.full(n, np.nan)
    weekly_s2 = np.full(n, np.nan)
    weekly_pp = np.full(n, np.nan)  # Pivot point
    
    # For each 6h bar, get the most recent completed weekly bar's OHLC
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        # Find the most recent completed 1w bar before current 6h bar
        prior_1w_bars = df_1w[df_1w['open_time'] < current_time]
        if len(prior_1w_bars) > 0:
            prev_week = prior_1w_bars.iloc[-1]
            ph = prev_week['high']
            pl = prev_week['low']
            pc = prev_week['close']
            
            # Weekly pivot formulas
            pivot_point = (ph + pl + pc) / 3
            weekly_pp[i] = pivot_point
            weekly_r1[i] = 2 * pivot_point - pl
            weekly_s1[i] = 2 * pivot_point - ph
            weekly_r2[i] = pivot_point + (ph - pl)
            weekly_s2[i] = pivot_point - (ph - pl)
        else:
            # Not enough prior data
            weekly_pp[i] = np.nan
            weekly_r1[i] = np.nan
            weekly_s1[i] = np.nan
            weekly_r2[i] = np.nan
            weekly_s2[i] = np.nan
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d_vol = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d_vol) >= 20:
        vol_1d = df_1d_vol['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d_vol, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
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
        if (np.isnan(weekly_r1[i]) or np.isnan(weekly_s1[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- ATR Expansion: Require current 1d ATR > 1.2x average ATR (volatility expansion) ---
        # Since we only have aligned ATR, we compare to its rolling average
        if i >= 50:  # Need enough history for ATR average
            atr_hist = atr_14_1d_aligned[max(0, i-49):i+1]
            atr_avg = np.nanmean(atr_hist[-20:]) if len(atr_hist) >= 20 else np.nanmean(atr_hist)
            atr_expansion = not np.isnan(atr_avg) and atr_14_1d_aligned[i] > 1.2 * atr_avg
        else:
            atr_expansion = True  # Allow early trades during warmup
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for 6h timeframe for stoploss
            tr_6h = np.zeros(i+1)
            tr_6h[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr_6h[j] = max(high[j] - low[j], 
                               abs(high[j] - close[j-1]),
                               abs(low[j] - close[j-1]))
            atr_14_6h = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14_6h
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at weekly R2 (strong resistance)
                if close[i] >= weekly_r2[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14_6h
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at weekly S2 (strong support)
                if close[i] <= weekly_s2[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above weekly R1 with ATR expansion and volume
        long_condition = (
            close[i] > weekly_r1[i] and 
            atr_expansion and 
            volume_spike
        )
        
        # Short: Price breaks below weekly S1 with ATR expansion and volume
        short_condition = (
            close[i] < weekly_s1[i] and 
            atr_expansion and 
            volume_spike
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