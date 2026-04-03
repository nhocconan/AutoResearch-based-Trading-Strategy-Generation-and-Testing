#!/usr/bin/env python3
"""
Experiment #075: 6h Williams %R reversal + 1w volume spike + 1d ADX trend filter

HYPOTHESIS: Williams %R (14) identifies overbought/oversold conditions on 6h timeframe.
Entries occur when %R reverses from extreme levels (< -80 for long, > -20 for short)
confirmed by 1w volume spike (> 2.0x average) and 1d ADX > 25 (trending market).
This captures mean reversion in strong trends while avoiding choppy markets.
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
Works in both bull/bear markets: in bull markets, longs from oversold; in bear markets, shorts from overbought.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_1d[0] - low_1d[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                           np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                            np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_period = 14
        atr = pd.Series(tr).ewm(span=tr_period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / np.where(atr != 0, atr, np.nan)
        di_minus = 100 * dm_minus_smooth / np.where(atr != 0, atr, np.nan)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) != 0, (di_plus + di_minus), np.nan)
        adx_14_1d = pd.Series(dx).ewm(span=tr_period, adjust=False).mean().values
        adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    else:
        adx_14_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for volume confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio (current vs 20-period average) on 1w
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0  # Neutral for warmup
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === 6h Indicators ===
    # Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / np.where((highest_high - lowest_low) != 0, (highest_high - lowest_low), np.nan)
    
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
        if (np.isnan(adx_14_1d_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Require ADX > 25 (trending market) ---
        trending = adx_14_1d_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) on 1w ---
        volume_spike = vol_ratio_1w_aligned[i] > 2.0
        
        # --- Williams %R Levels ---
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # --- Williams %R Reversal Detection ---
        # Long reversal: %R was oversold and now rising
        long_reversal = oversold and (i == warmup or williams_r[i] > williams_r[i-1])
        # Short reversal: %R was overbought and now falling
        short_reversal = overbought and (i == warmup or williams_r[i] < williams_r[i-1])
        
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
                # Take profit at Williams %R > -50 (exit oversold)
                if williams_r[i] > -50:
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
                # Take profit at Williams %R < -50 (exit overbought)
                if williams_r[i] < -50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Williams %R reversal from oversold with volume and trend
        long_condition = (
            long_reversal and 
            volume_spike and 
            trending
        )
        
        # Short: Williams %R reversal from overbought with volume and trend
        short_condition = (
            short_reversal and 
            volume_spike and 
            trending
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

</think>