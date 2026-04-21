#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_WeeklyPivotDirection_VolumeConfirmation_v1
Hypothesis: Use 1w pivot direction (bullish/bearish) to filter 6h Camarilla(R1/S1) breakouts.
In weekly bullish context (price above weekly pivot), only long breakouts at R1 are taken.
In weekly bearish context (price below weekly pivot), only short breakdowns at S1 are taken.
Adds 6h volume confirmation (>1.5x 20-period average) to ensure participation.
Uses discrete sizing (0.25) and ATR-based stoploss (2.0x ATR) to manage risk.
Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
Works in bull/bear via weekly pivot regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for pivot direction, 1d for volume regime context)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1w pivot direction: bullish if close > pivot, bearish if close < pivot ===
    # Using previous completed 1w bar's OHLC to calculate pivot (no look-ahead)
    prev_1w_high = np.roll(df_1w['high'].values, 1)
    prev_1w_low = np.roll(df_1w['low'].values, 1)
    prev_1w_close = np.roll(df_1w['close'].values, 1)
    prev_1w_high[0] = prev_1w_low[0] = prev_1w_close[0] = np.nan
    
    weekly_pivot = (prev_1w_high + prev_1w_low + prev_1w_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Weekly close price for direction (aligned)
    weekly_close = prev_1w_close  # already shifted
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    # Weekly bullish/bearish regime
    weekly_bullish = weekly_close_aligned > weekly_pivot_aligned
    weekly_bearish = weekly_close_aligned < weekly_pivot_aligned
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (>1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    # === 6h Camarilla pivot levels (R1, S1) based on PREVIOUS bar's OHLC ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar invalid
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(volume_confirmed[i]) or 
            np.isnan(weekly_bullish[i]) or np.isnan(weekly_bearish[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed[i]
        wb = weekly_bullish[i]
        wbear = weekly_bearish[i]
        
        if position == 0:
            # Only trade in direction of weekly pivot
            # Weekly bullish: look for longs at R1 breakout
            # Weekly bearish: look for shorts at S1 breakdown
            long_condition = wb and (price > r1_val) and vol_conf
            short_condition = wbear and (price < s1_val) and vol_conf
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below S1 (failed breakout)
                elif price < s1_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks above R1 (failed breakdown)
                elif price > r1_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_WeeklyPivotDirection_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0