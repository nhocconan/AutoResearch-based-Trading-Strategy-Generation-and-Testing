#!/usr/bin/env python3
"""
Experiment #431: 6h Donchian Breakout + 1d Weekly Pivot + Volume Confirmation

HYPOTHESIS: Donchian(20) breakouts on 6h timeframe, filtered by 1d weekly pivot levels 
(R1/S1 for continuation, R2/S2 for reversal) and 6h volume spike (>2x average), creates 
a robust strategy for both bull and bear markets. Weekly pivots provide institutional 
reference points, Donchian channels capture breakouts, and volume confirms participation. 
Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while 
capitalizing on high-probability breakouts at key weekly levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot levels from prior week's OHLC
    weekly_pivot = np.full(n, np.nan)
    weekly_r1 = np.full(n, np.nan)
    weekly_s1 = np.full(n, np.nan)
    weekly_r2 = np.full(n, np.nan)
    weekly_s2 = np.full(n, np.nan)
    
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        # Find the most recent completed week (7d) before current 6h bar
        # Use 1d data: find bars with open_time < current_time and spaced ~7 days apart
        prior_1d_bars = df_1d[df_1d['open_time'] < current_time]
        if len(prior_1d_bars) >= 7:
            # Get the bar from ~7 days ago (prior week's equivalent time)
            week_ago_idx = len(prior_1d_bars) - 7
            if week_ago_idx >= 0:
                week_bar = prior_1d_bars.iloc[week_ago_idx]
                ph = week_bar['high']
                pl = week_bar['low']
                pc = week_bar['close']
                
                # Weekly pivot calculations (standard floor trader's method)
                pivot = (ph + pl + pc) / 3
                weekly_pivot[i] = pivot
                weekly_r1[i] = 2 * pivot - pl
                weekly_s1[i] = 2 * pivot - ph
                weekly_r2[i] = pivot + (ph - pl)
                weekly_s2[i] = pivot - (ph - pl)
        else:
            # Not enough prior data
            weekly_pivot[i] = np.nan
            weekly_r1[i] = np.nan
            weekly_s1[i] = np.nan
            weekly_r2[i] = np.nan
            weekly_s2[i] = np.nan
    
    # === HTF: 6h data for volume spike (Call ONCE before loop) ===
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate volume ratio (current vs 20-period average) on 6h
    if len(df_6h) >= 20:
        vol_6h = df_6h['volume'].values
        vol_ma_20 = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_6h = np.zeros(len(vol_6h))
        vol_ratio_6h[20:] = vol_6h[20:] / vol_ma_20[20:]
        vol_ratio_6h[:20] = 1.0  # Neutral for warmup
        vol_ratio_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio_6h)
    else:
        vol_ratio_6h_aligned = np.full(n, 1.0)
    
    # === 6h Indicators: Donchian Channel (20) ===
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 20:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
        else:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot[i]) or np.isnan(weekly_r1[i]) or np.isnan(weekly_s1[i]) or
            np.isnan(weekly_r2[i]) or np.isnan(weekly_s2[i]) or np.isnan(vol_ratio_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_6h_aligned[i] > 2.0
        
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
                # Take profit at weekly R2 (strong resistance) or reverse at S1
                if close[i] >= weekly_r2[i] or close[i] <= weekly_s1[i]:
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
                # Take profit at weekly S2 (strong support) or reverse at R1
                if close[i] <= weekly_s2[i] or close[i] >= weekly_r1[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Donchian high with volume, above weekly pivot (bullish bias)
        long_condition = (
            close[i] > donchian_high[i] and  # Donchian breakout
            volume_spike and                 # Volume confirmation
            close[i] > weekly_pivot[i]       # Above weekly pivot (bullish bias)
        )
        
        # Short: Break below Donchian low with volume, below weekly pivot (bearish bias)
        short_condition = (
            close[i] < donchian_low[i] and   # Donchian breakdown
            volume_spike and                 # Volume confirmation
            close[i] < weekly_pivot[i]       # Below weekly pivot (bearish bias)
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